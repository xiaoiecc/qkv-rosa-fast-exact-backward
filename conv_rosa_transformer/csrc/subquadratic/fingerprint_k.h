#pragma once
// Exact K credit from bounded, fingerprint-reported full-window witnesses.
// Fingerprints only report endpoints; every reported repair uses exact LCS.
#include "index.h"
#include "../full_word_map.h"
#include <cmath>
#include <memory>

namespace rosa_subq {
namespace fingerprint_detail {
inline std::uint64_t mix(std::uint64_t x) {
    x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
    return x^(x>>31);
}
inline int bit_index(std::uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#else
    int bit=0;while(!(x&1)){x>>=1;++bit;}return bit;
#endif
}
struct Dictionary {
    rosa_exact::FullWordMap classes;
    std::vector<int> offsets,ends;
    Dictionary(int n,int length,const std::vector<std::uint64_t>&prefix,
            std::uint64_t power,std::uint64_t seed,std::uint64_t mask)
      :classes(n,seed) {
        std::vector<int> group(n,-1),count;
        for(int end=length-1;end<n;++end) {
            const auto key=(prefix[end+1]-prefix[end+1-length]*power)&mask;
            const auto found=classes.insert(key,int(count.size()));
            if(found.second)count.push_back(0);
            group[end]=found.first;++count[found.first];
        }
        offsets.resize(count.size()+1);
        for(std::size_t c=0;c<count.size();++c)offsets[c+1]=offsets[c]+count[c];
        ends.resize(n-length+1);count=offsets;
        for(int end=length-1;end<n;++end)ends[count[group[end]]++]=end;
    }
    std::size_t bytes() const {
        return classes.workspace_bytes()+(offsets.capacity()+ends.capacity())*sizeof(int);
    }
};

// mask exists only to exercise forced fingerprint collisions in the oracle.
// The public wrapper always uses every fingerprint bit.
inline bool run(const Index& index,const Scores& scores,int bits,
        std::vector<double>& credit,Stats& stats,std::uint64_t budget,
        bool include_deletion,std::uint64_t seed,std::uint64_t mask) {
    const int n=index.n,d=scores.d;
    if(bits<1 || bits>64 || scores.n!=n || d<1 || !scores.weights ||
       !scores.payload || !scores.empty || credit.size()!=std::size_t(n)*bits)
        throw std::invalid_argument("invalid fingerprint K dimensions");
    int logarithm=0;for(unsigned x=static_cast<unsigned>(n);x;x>>=1)++logarithm;
    const int cap=std::max(32,2*bits+2*logarithm);
    int longest=0;for(const Route route:index.baseline())longest=std::max(longest,route.length);
    if(longest>cap)return false;
    const int maximum=std::max(1,longest);
    const std::uint64_t allowed=bits==64?UINT64_MAX:(UINT64_C(1)<<bits)-1;
    constexpr std::uint64_t multiplier=UINT64_C(0x9e3779b185ebca87);
    // Candidate sets and budget acceptance must not change with the runtime
    // placement seed. Only FullWordMap placement below uses that seed.
    constexpr std::uint64_t symbol_salt=UINT64_C(0xa0761d6478bd642f);
    std::vector<std::uint64_t> powers(std::size_t(maximum)+1,1),qp(std::size_t(n)+1),kp(qp.size());
    std::vector<std::uint64_t> replacement(std::size_t(n)*bits);
    for(int m=1;m<=maximum;++m)powers[m]=powers[m-1]*multiplier;
    rosa_exact::FullWordMap first_symbol(n,seed);
    for(int t=0;t<n;++t)first_symbol.insert(index.k[t],t);
    for(int t=0;t<n;++t) {
        const auto q=mix(index.q[t]^symbol_salt),k=mix(index.k[t]^symbol_salt);
        qp[t+1]=qp[t]*multiplier+q;kp[t+1]=kp[t]*multiplier+k;
        for(int bit=0;bit<bits;++bit) {
            const auto changed=index.q[t]^(UINT64_C(1)<<bit);const int first=first_symbol.find(changed);
            // A K center corresponding to Q owner t must be strictly before t.
            // mix is a permutation, so zero unambiguously marks no support.
            replacement[std::size_t(t)*bits+bit]=first>=0 && first<t?mix(changed^symbol_salt)-q:0;
        }
    }
    std::vector<std::unique_ptr<Dictionary>> dictionaries(std::size_t(maximum)+1);
    std::vector<int> visited(n,-1),latest(std::size_t(maximum)+1,-1);
    std::vector<Route> winners(credit.size()),deleted(maximum);
    std::vector<std::size_t> touched;
    std::vector<double> delta(credit.size(),0.),deletion_delta(include_deletion?n:0,0.);
    std::uint64_t work=0;Stats pending=stats;std::size_t dictionary_bytes=0;
    const auto spend=[&]() {if(work==budget)return false;++work;return true;};
    const auto local_lcs=[&](int a,int b) {
        if(a<0 || b<0)return 0;
        assert(b<a);
        // All arms remain causal. Their length is at most baseline[a].L,
        // already certified <=cap=O(bits+log n), so this scan is bounded.
        const int limit=std::min(b+1,index.baseline()[a].length);int matched=0;
        while(matched<limit && index.q[a-matched]==index.k[b-matched])++matched;
        return matched;
    };
    const auto dictionary_for=[&](int m)->const Dictionary& {
        if(!dictionaries[m]) {
            dictionaries[m].reset(new Dictionary(n,m,kp,powers[m],seed^std::uint64_t(m),mask));
            dictionary_bytes+=dictionaries[m]->bytes();
        }
        return *dictionaries[m];
    };
    for(int t=1;t<n;++t) {
        const double* weight=scores.row(t);
        if(std::none_of(weight,weight+d,[](double x){return x!=0.;}))continue;
        const Route original=index.baseline()[t];const int length=original.length;
        const int first=original.end-length+1,m=std::max(1,length);
        const auto& dictionary=dictionary_for(m);touched.clear();
        const auto consider=[&](int end) {
            assert(end>=0 && end<t);
            if(visited[end]==t)return;
            visited[end]=t;
            const int right=local_lcs(t,end);++pending.suffix_queries;
            const int owner=end-right,center=t-right;if(owner<0)return;
            const auto difference=index.q[center]^index.k[owner];
            if(!difference || (difference&(difference-1)) || (difference&~allowed))return;
            const bool affected=length && owner>=first && owner<=original.end;
            const Route surviving=affected?(owner<original.end?Route{original.end-owner,original.end}:Route{}):original;
            if(right+1<surviving.length) {
                const int start=end-surviving.length+1;
                if(start<0 || index.q[t-surviving.length+1]!=index.k[start])return;
            }
            const int left=local_lcs(center-1,owner-1);++pending.suffix_queries;
            const Route candidate{right+1+left,end};if(!(candidate>surviving))return;
            const auto key=std::size_t(owner)*bits+bit_index(difference);
            if(!winners[key].length)touched.push_back(key);
            winners[key]=std::max(winners[key],candidate);++pending.k_centers;
        };
        const auto query=[&](std::uint64_t key) {
            if(!spend())return false;
            const int group=dictionary.classes.find(key&mask);if(group<0)return true;
            for(int at=dictionary.offsets[group];at<dictionary.offsets[group+1];++at) {
                const int end=dictionary.ends[at];
                // Charge before causality and visited filters. Fingerprint
                // collisions can only spend this finite report budget.
                if(!spend())return false;
                if(end>=t)break;
                consider(end);
            }
            return true;
        };
        const auto original_hash=qp[t+1]-qp[t+1-m]*powers[m];
        if(!query(original_hash))return false;
        for(int offset=0;offset<m;++offset) {
            const auto* changes=replacement.data()+std::size_t(t-offset)*bits;
            for(int bit=0;bit<bits;++bit)
                if(changes[bit] && !query(original_hash+changes[bit]*powers[offset]))return false;
        }
        if(length)for(int end=first;end<std::min(t,original.end+length);++end) {
            if(!spend())return false;consider(end);
        }

        // Reuse the same verified fingerprint index for exact occurrence
        // predecessors. A collision only adds backwards candidates to test.
        const auto exact_R=[&](int h,int cutoff) {
            if(!spend())return -2;
            if(h>cutoff)return -1;
            const auto& words=dictionary_for(h);
            const auto hash=(qp[t+1]-qp[t+1-h]*powers[h])&mask;
            const int group=words.classes.find(hash);if(group<0)return -1;
            const auto begin=words.ends.begin()+words.offsets[group];
            auto at=std::lower_bound(begin,words.ends.begin()+words.offsets[group+1],cutoff);
            while(at!=begin) {
                if(!spend())return -2;
                const int end=*--at;++pending.suffix_queries;
                if(local_lcs(t,end)>=h)return end;
            }
            return -1;
        };
        // R[h]-h strictly decreases with h. For increasing deleted owner s,
        // the largest surviving right height therefore only decreases. The
        // left winner grows by one newly admitted endpoint at each owner.
        if(length) {
            int h=1;
            while(h<=length) {
                const int end=exact_R(h,t);if(end==-2)return false;
                const int high=end==original.end?length:std::min(length,local_lcs(t,end));
                if(end!=original.end)++pending.suffix_queries;
                assert(end>=0 && high>=h);
                std::fill(latest.begin()+h,latest.begin()+high+1,end);h=high+1;
            }
            Route prefix;int low=1,high=std::min(length,first);
            while(low<=high) {
                const int mid=low+(high-low)/2,end=exact_R(mid,first);if(end==-2)return false;
                if(end<0)high=mid-1;
                else {prefix={mid,end};low=mid+1;}
            }
            h=length;
            for(int owner=first;owner<=original.end;++owner) {
                if(owner>first) {
                    const int l=local_lcs(t,owner-1);++pending.suffix_queries;
                    if(l)prefix=std::max(prefix,Route{l,owner-1});
                }
                while(h && latest[h]-h<owner)--h;
                deleted[owner-first]=std::max(prefix,h?Route{h,latest[h]}:Route{});
                if(include_deletion && deleted[owner-first].end!=original.end) {
                    const auto* after=scores.value(deleted[owner-first].end);
                    const auto* before=scores.value(original.end);double value=0.;
                    for(int c=0;c<d;++c)value+=weight[c]*(after[c]-before[c]);
                    deletion_delta[owner]+=value;
                }
            }
        }
        for(const auto key:touched) {
            const auto candidate=winners[key];winners[key]={};
            const int owner=int(key/std::size_t(bits));
            const Route removed=length && owner>=first && owner<=original.end?deleted[owner-first]:original;
            if(!(candidate>removed) || candidate.end==removed.end)continue;
            const auto* after=scores.value(candidate.end);const auto* before=scores.value(removed.end);double value=0.;
            for(int c=0;c<d;++c)value+=weight[c]*(after[c]-before[c]);
            delta[key]+=value;
        }
    }
    for(int owner=0;owner<n;++owner)for(int bit=0;bit<bits;++bit) {
        const auto key=std::size_t(owner)*bits+bit;credit[key]+=delta[key];
        if(include_deletion)credit[key]+=deletion_delta[owner];
    }
    const std::size_t bytes=dictionary_bytes+first_symbol.workspace_bytes()+std::size_t(2*n)*sizeof(int)+dictionaries.capacity()*sizeof(std::unique_ptr<Dictionary>)
        +(powers.capacity()+qp.capacity()+kp.capacity()+replacement.capacity())*sizeof(std::uint64_t)
        +(visited.capacity()+latest.capacity())*sizeof(int)+(winners.capacity()+deleted.capacity())*sizeof(Route)
        +touched.capacity()*sizeof(std::size_t)+(delta.capacity()+deletion_delta.capacity())*sizeof(double);
    pending.peak_workspace_bytes=std::max<std::uint64_t>(pending.peak_workspace_bytes,index.workspace_bytes()+bytes);
    stats=pending;return true;
}
} // namespace fingerprint_detail
inline bool fingerprint_k(const Index& index,const Scores& scores,int bits,
        std::vector<double>& credit,Stats& stats,std::uint64_t budget,
        bool include_deletion=false,std::uint64_t seed=0) {
    return fingerprint_detail::run(index,scores,bits,credit,stats,budget,include_deletion,seed,UINT64_MAX);
}
} // namespace rosa_subq
