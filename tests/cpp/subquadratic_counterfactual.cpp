// Compare complete credits to physically flipping the input and rerunning a
// causal suffix DP. No production matching/index routine builds the reference.
#include "../../conv_rosa_transformer/csrc/subquadratic/endpoint_credit.h"
#include "../../conv_rosa_transformer/csrc/subquadratic/packed_context.h"
#include "../../conv_rosa_transformer/csrc/subquadratic/run_credit.h"
#include "../../conv_rosa_transformer/csrc/subquadratic/random_context.h"
#include <iostream>
#include <random>

using namespace rosa_subq;
using Words=std::vector<uint64_t>;
using Values=std::vector<double>;
int cases=0,context_accepted=0,context_deep=0,context_budget_stops=0,context_length_stops=0;
uint64_t components=0;
std::mt19937_64 rng(2026112204);

std::vector<Route> literal(const Words& q,const Words& k) {
    const int n=int(q.size());
    std::vector<int> length(n);
    std::vector<Route> result(n);
    for(int t=1;t<n;++t)for(int r=t-1;r>=0;--r) {
        length[r]=q[t]==k[r]?1+(r?length[r-1]:0):0;
        if(length[r])result[t]=std::max(result[t],Route{length[r],r});
    }
    return result;
}
void compare(const char* method,const Values& actual,const Values& expected) {
    for(size_t i=0;i<actual.size();++i) {
        if(std::abs(actual[i]-expected[i])>1e-8 || !std::isfinite(actual[i])) {
            std::cerr<<method<<" case "<<cases<<" scalar "<<i<<" got "<<actual[i]<<" expected "<<expected[i]<<'\n';
            throw std::runtime_error("credit mismatch");
        }
        ++components;
    }
}
void check(Words q,Words k,int bits,bool test_residual=false) {
    const int n=int(q.size()),d=3;
    const auto base=literal(q,k);
    Values w(n*d),v(n*d),empty(d);
    for(auto* a:{&w,&v,&empty})for(auto& x:*a)x=double(int(rng()%19)-9)/8.;
    Scores score{n,d,w.data(),v.data(),empty.data()};
    Values eq(n*bits),ek(n*bits);
    Values gradient(n*bits),diagonal(n*bits),empty_value(bits),scale(bits),equal_q(n*bits),equal_k(n*bits);
    std::vector<uint8_t> value(n*bits);
    std::vector<int32_t> rows(n),ends(n);
    std::iota(rows.begin(),rows.end(),0);
    for(int t=0;t<n;++t)ends[t]=base[t].end;
    for(auto* a:{&gradient,&diagonal,&empty_value,&scale})for(auto& x:*a)x=double(int(rng()%7)-3)/4.;
    for(auto& h:diagonal)h=std::abs(h);
    for(auto& x:value)x=rng()%2;
    const bool quadratic=cases%2==0;
    auto payload=[&](int end,int j){return end<0?empty_value[j]:(2*int(value[(end+1)*bits+j])-1)*scale[j];};
    for(int side=0;side<2;++side)for(int owner=0;owner<n;++owner)for(int bit=0;bit<bits;++bit) {
        auto& word=side?k[owner]:q[owner]; word^=UINT64_C(1)<<bit;
        const auto flip=literal(q,k); word^=UINT64_C(1)<<bit;
        double sum=0;
        for(int t=0;t<n;++t)sum+=score.score(t,flip[t].end)-score.score(t,base[t].end);
        double projected=0;
        for(int t=0;t<n;++t)if(flip[t].end!=base[t].end)for(int j=0;j<bits;++j) {
            const double dy=payload(flip[t].end,j)-payload(base[t].end,j);
            projected+=gradient[t*bits+j]*dy+(quadratic?.5*diagonal[t*bits+j]*dy*dy:0.);
        }
        (side?equal_k:equal_q)[owner*bits+bit]=projected;
        (side?ek:eq)[owner*bits+bit]=sum;
    }
    {
        rosa_random::OnegramPlan plan(q.data(),k.data(),n,bits);
        const uint64_t exact=endpoint_credit::equal_pairs(plan,UINT64_MAX);
        uint64_t expected_count=0;
        for(int t=1;t<n;++t)for(int r=0;r<t;++r)expected_count+=q[t]==k[r];
        if(exact!=expected_count)throw std::runtime_error("incorrect exact join size");
        Values a(n*bits),b(n*bits);Stats st;
        endpoint_credit::equal_sweep(plan,exact,base,rows,value.data(),gradient.data(),
            quadratic?diagonal.data():nullptr,empty_value.data(),scale.data(),ends.data(),false,0,a,b,st);
        compare("equality Q",a,equal_q);compare("equality K",b,equal_k);
    }
    auto sweep=[&](const std::vector<int32_t>& rows,Values& cq,Values& ck) {
        Stats st;
        Index index(q.data(),k.data(),n,false,0,&base);
        endpoint_credit::RowSweep sorted(index,bits,st);
        endpoint_credit::Occurrences occ(k.data(),n,bits);
        if(!occ.admit(q.data(),k.data(),n,UINT64_MAX))throw std::runtime_error("occurrence census");
        Values qdiff(n+1),kdiff(n+1);
        for(int t:rows) {
            auto cost=[&](int r){return score.score(t,r)-score.score(t,base[t].end);};
            sorted.row(t,[&](auto emit){occ.row(q[t],t,emit);},cost,qdiff,kdiff,cq,ck);
        }
        double a=0,b=0;
        for(int i=0;i<n;++i) {
            a+=qdiff[i];b+=kdiff[i];
            for(int j=0;j<bits;++j){cq[i*bits+j]+=a;ck[i*bits+j]+=b;}
        }
    };
    Values cq(n*bits),ck(n*bits);
    sweep(rows,cq,ck);
    compare("endpoint Q",cq,eq);compare("endpoint K",ck,ek);
    int longest=0;for(auto r:base)longest=std::max(longest,r.length);
    {
        // Start with nonzero output buffers to detect partial commits on
        // rejection. A successful call returns the complete credit.
        Values a(n*bits,123.),b(n*bits,-45.);
        Stats st;
        const bool accepted=random_context::credit(q.data(),k.data(),base,bits,longest,score,
                                                   a,b,st,UINT64_MAX);
        if(accepted!=(longest<=random_context::max_length(n,bits)))
            throw std::runtime_error("unexpected context admission");
        if(accepted) {
            ++context_accepted;
            if(longest>int(std::ceil(std::log2(double(n))/bits)))++context_deep;
            compare("causal trie Q",a,eq);compare("causal trie K",b,ek);
        } else {
            compare("length guard Q",a,Values(n*bits,123.));
            compare("length guard K",b,Values(n*bits,-45.));
            if(st.suffix_queries || st.peak_workspace_bytes)throw std::runtime_error("length guard statistics");
            ++context_length_stops;
        }
        // Fail on the final charged report, after previous rows have
        // accumulated their contributions in the private working arrays.
        if(accepted && st.suffix_queries) {
            a.assign(n*bits,123.);b.assign(n*bits,-45.);
            const uint64_t cap=st.suffix_queries-1;
            Stats stopped;stopped.suffix_queries=11;stopped.peak_workspace_bytes=29;
            if(random_context::credit(q.data(),k.data(),base,bits,longest,score,a,b,stopped,cap))
                throw std::runtime_error("exceeded trie report budget");
            if(stopped.suffix_queries!=11 || stopped.peak_workspace_bytes!=29)
                throw std::runtime_error("trie budget guard statistics");
            compare("trie budget Q",a,Values(n*bits,123.));
            compare("trie budget K",b,Values(n*bits,-45.));
            ++context_budget_stops;
        }
    }
    if(PackedContext::fits(bits,longest) && std::all_of(q.begin(),q.end(),[&](auto x){return !(x>>bits);})
            && std::all_of(k.begin(),k.end(),[&](auto x){return !(x>>bits);})) {
        Stats st;
        if(!packed_credit(q.data(),k.data(),base,bits,longest,score,cq,ck,st,UINT64_MAX))
            throw std::runtime_error("unbounded packed credit rejected");
        compare("packed Q",cq,eq);compare("packed K",ck,ek);
        if(n>1 && (packed_credit(q.data(),k.data(),base,bits,longest,score,cq,ck,st,0)))
            throw std::runtime_error("accepted zero candidate budget");
        compare("aborted Q",cq,eq);compare("aborted K",ck,ek);
    }
    if(test_residual || std::all_of(q.begin()+1,q.end(),[&](auto x){return x==q[1];})) {
        const auto word=q[1];Words uniform(n,word);
        auto filtered=w;auto subset=score;subset.weights=filtered.data();
        std::vector<int32_t> residual;
        if(!run_credit::residual_rows(q.data(),k.data(),n,word,rows,residual)
                || (test_residual && residual.empty()))throw std::runtime_error("unexpected mixed certificate");
        for(int t:residual)std::fill_n(filtered.begin()+t*d,d,0.);
        cq.assign(n*bits,0.);ck.assign(n*bits,0.);Stats st;
        run_credit::evaluate(k.data(),word,literal(uniform,k),subset,bits,cq,ck,st);
        sweep(residual,cq,ck);
        compare("run Q",cq,eq);compare("run K",ck,ek);
    }
    ++cases;
}
int main() {
    for(int n=2;n<=5;++n)for(int a=0;a<(1<<n);++a)for(int b=0;b<(1<<n);++b) {
        Words q(n),k(n);
        for(int i=0;i<n;++i){q[i]=(a>>i)&1;k[i]=(b>>i)&1;}
        check(q,k,1);
    }
    for(int n=6;n<=8;++n)for(int b=0;b<(1<<n);++b)for(int x=0;x<=1;++x) {
        Words q(n,x),k(n);q[0]=1-x;
        for(int i=0;i<n;++i)k[i]=(b>>i)&1;
        check(q,k,2);
    }
    for(int bits:{1,2,3,4,5,7,8,9,12,16,31,32,63,64})for(int rep=0;rep<10;++rep) {
        const int n=9+rep;Words q(n),k(n);const uint64_t mask=bits==64?UINT64_MAX:(UINT64_C(1)<<bits)-1;
        for(auto& x:q)x=rng()&mask;
        for(int i=0;i<n;++i)k[i]=(rep%2?q[(i+3)%n]:rng())&mask;
        k[3]^=UINT64_C(1)<<(bits-1);check(q,k,bits);
    }
    // Three exact symbols cross the one-level directory on width 9.
    // The competing copy has the same terminal symbol but a distinct
    // prefix, and extending a one-bit repair can change the winner.
    {
        Words q(48),k(48);
        for(int i=0;i<48;++i){q[i]=200+i;k[i]=300+i;}
        q[22]=k[3]=k[9]=11;q[23]=k[4]=k[10]=26;q[24]=k[5]=k[11]=33;
        q[25]=39;k[6]=38;k[12]=37;
        check(q,k,9);
    }
    // Forbidding a K owner erases several overlapping occurrences.
    // The stored list must reach the best earlier surviving endpoint.
    {
        Words q(32),k(32);
        for(int i=0;i<32;++i){q[i]=(i%3)+1;k[i]=(i%5)+1;}
        for(int i=0;i<7;++i)q[17+i]=k[7+i]=0;
        check(q,k,3);
    }
    // Bit 63 appears in both data and edits. Keep the baseline short
    // enough for the width-64 path to execute, including latest-end ties.
    {
        const uint64_t top=UINT64_C(1)<<63;
        check({13,top,13,top+3,5,top,7,top+5},
              {0,9,top+3,4,top+1,7,6,top},64);
    }
    Words q(96),k(96);q[28]=1;q[75]=2;
    for(int i=0;i<96;++i)k[i]=i%3?rng()%4:3;
    check(q,k,2,true);
    for(int n:{31,32,33,34})check(Words(n,0),Words(n,0),1);
    check({9,9,12,3,9,3},{9,12,9,10,3,2},2);
    // Force collisions in the projected dictionary, then check the exact census.
    q.resize(200);k.resize(200);
    for(int i=0;i<200;++i)q[i]=k[i]=uint32_t(rng());
    std::rotate(k.begin(),k.begin()+3,k.end());
    endpoint_credit::Reports report;
    if(report.build(q.data(),k.data(),200,64))throw std::runtime_error("collision budget");
    endpoint_credit::Occurrences occ(k.data(),200,64);
    if(!occ.admit(q.data(),k.data(),200,2400))throw std::runtime_error("sparse exact census");
    for(int t=0;t<200;++t) {
        std::vector<int> got,want;
        occ.row(q[t],t,[&](int r){got.push_back(r);});
        for(int r=0;r<t;++r){const auto x=q[t]^k[r];if(!(x&(x-1)))want.push_back(r);}
        std::sort(got.begin(),got.end());
        if(got!=want)throw std::runtime_error("causal report mismatch");
    }
    for(int owners:{3,16,53})for(int length:{1,17,77})for(int pad:{0,7}) {
        const int d=3,available=owners+length-1-pad;
        if(available<1)continue;
        Values signal(available*d),kernel(length*d),actual(owners,2.),expected=actual;
        for(auto* a:{&signal,&kernel})for(auto& x:*a)x=double(int(rng()%128)-64)/13.;
        for(int u=0;u<owners;++u)for(int h=0;h<length && u+h<available;++h)
            for(int c=0;c<d;++c)expected[u]+=signal[(u+h)*d+c]*kernel[h*d+c];
        CorrelationStats st;
        BlockCorrelations::middle_product(signal.data(),available,d,kernel.data(),length,owners,actual.data(),st);
        compare("middle product",actual,expected);
    }
    // Both repairs share the largest suffix of whole symbols that fits in
    // one 64-bit word and break at the same edit.
    // Only the earlier key copy extends farther in the preceding machine
    // word. Truncating to one word would choose the later endpoint.
    for(int bits=1;bits<=8;++bits) {
        const int digits=64/bits,extra=std::max(2,digits/8);
        const int longer=digits+extra,shorter=digits+1,later=longer+5;
        const int n=3*longer+7,t=n-1,owner=t-digits/2;
        const uint64_t mask=(UINT64_C(1)<<bits)-1;
        Words q(n),k(n);
        for(auto& w:q)w=rng()&mask;
        for(auto& w:k)w=rng()&mask;
        std::copy_n(q.end()-longer,longer,k.begin());
        std::copy_n(q.end()-shorter,shorter,k.begin()+later);
        k[later-1]=q[t-shorter]^1;
        if(literal(q,k)[t]!=Route{longer,longer-1})throw std::runtime_error("packed boundary fixture");
        q[owner]^=1;
        auto before=literal(q,k);
        int longest=0;for(Route route:before)longest=std::max(longest,route.length);
        if(!PackedContext::fits(bits,longest))throw std::runtime_error("packed boundary rejected");
        check(q,k,bits);
    }
    if(!context_deep || !context_length_stops || !context_budget_stops)
        throw std::runtime_error("uncovered trie path");
    std::cout<<"{\"context_accepted\":"<<context_accepted<<",\"context_deep\":"<<context_deep
             <<",\"context_budget_stops\":"<<context_budget_stops
             <<",\"context_length_stops\":"<<context_length_stops<<",\"status\":\"PASS\",\"stream_pairs\":"<<cases<<",\"compared_scalars\":"<<components<<"}\n";
}
