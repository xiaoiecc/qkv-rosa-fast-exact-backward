// Independent literal edited-string oracle for the compact short K kernel.
// Build: c++ -O2 -std=c++17 tests/cpp/subquadratic_short.cpp -o /tmp/rosa-short
#include "../../conv_rosa_transformer/csrc/subquadratic/short_k.h"
#include "../../conv_rosa_transformer/csrc/subquadratic/near_k.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>

using namespace rosa_subq;

static Route literal_route(const std::vector<std::uint64_t>& q,const std::vector<std::uint64_t>& k,int t,int forbidden=-1) {
    Route best;
    for(int r=0;r<t;++r) {
        int length=0;
        while(length<=t && length<=r && r-length!=forbidden && q[t-length]==k[r-length])++length;
        if(length && Route{length,r}>best)best=Route{length,r};
    }
    return best;
}

static std::uint64_t cases=0,values=0,route_checks=0,dominance_checks=0;
static double worst_error=0;

static void check(const std::vector<std::uint64_t>& q,const std::vector<std::uint64_t>& k,
                  int bits,int dimensions,int cutoff,std::mt19937_64& rng,bool real=false) {
    const int n=int(q.size());
    std::vector<double> gradient(std::size_t(n)*dimensions),payload(gradient.size()),empty(dimensions);
    const auto value=[&](){return real?std::uniform_real_distribution<double>(-3,3)(rng):double(int(rng()%7)-3);};
    for(double& x:gradient)x=value();
    for(double& x:payload)x=value();
    for(double& x:empty)x=value();
    if(cases%5==0)for(int t=0;t<n;t+=2)std::fill_n(gradient.data()+std::size_t(t)*dimensions,dimensions,0.0);
    if(cases%23==0)std::fill(gradient.begin(),gradient.end(),0.0);
    Scores scores{n,dimensions,gradient.data(),payload.data(),empty.data()};
    Index index(q.data(),k.data(),n);Stats stats;
    std::vector<double> got(std::size_t(n)*bits,0.0),expected(got.size(),0.0);
    short_k(index,scores,bits,cutoff,got,stats);
    std::vector<double> near(got.size()),expected_near(got.size());
    if(!near_k(index,scores,bits,near,stats,std::numeric_limits<std::uint64_t>::max())) {
        std::cerr<<"unbounded near repair rejected\n";std::exit(1);
    }
    std::vector<double> full(got.size()),expected_full(got.size());
    near_k(index,scores,bits,full,stats,std::numeric_limits<std::uint64_t>::max(),true);
    std::vector<Route> original(n);for(int t=0;t<n;++t)original[t]=literal_route(q,k,t);
    auto edited=k;
    for(int s=0;s<n;++s)for(int bit=0;bit<bits;++bit) {
        edited[s]^=std::uint64_t(1)<<bit;
        for(int t=s+1;t<n;++t) {
            const Route actual=literal_route(q,edited,t),deleted=literal_route(q,k,t,s);++route_checks;
            const double* after=scores.value(actual.end);const double* before=scores.value(original[t].end);
            double full_term=0.;for(int c=0;c<dimensions;++c)full_term+=scores.row(t)[c]*(after[c]-before[c]);
            expected_full[std::size_t(s)*bits+bit]+=full_term;
            if(actual.length>0 && actual>deleted) {
                const double* a=scores.value(actual.end);const double* b=scores.value(deleted.end);
                double term=0;for(int c=0;c<dimensions;++c)term+=scores.row(t)[c]*(a[c]-b[c]);
                expected_near[std::size_t(s)*bits+bit]+=term;
                if(actual.length<=cutoff)expected[std::size_t(s)*bits+bit]+=term;
            }
        }
        edited[s]^=std::uint64_t(1)<<bit;
    }
    for(std::size_t i=0;i<got.size();++i) {
        const double full_error=std::abs(full[i]-expected_full[i]);worst_error=std::max(worst_error,full_error);
        if(full_error>(real?2e-10:0.0)) {
            std::cerr<<"near full credit mismatch case="<<cases<<" n="<<n<<" bits="<<bits
                     <<" owner="<<i/bits<<" bit="<<i%bits<<" got="<<full[i]<<" want="<<expected_full[i]<<'\n';
            std::exit(1);
        }
        const double near_error=std::abs(near[i]-expected_near[i]);worst_error=std::max(worst_error,near_error);
        if(near_error>(real?2e-10:0.0)) {
            std::cerr<<"near repair mismatch case="<<cases<<" n="<<n<<" bits="<<bits
                     <<" owner="<<i/bits<<" bit="<<i%bits<<" got="<<near[i]<<" want="<<expected_near[i]<<'\n';
            std::exit(1);
        }
        const double error=std::abs(got[i]-expected[i]);worst_error=std::max(worst_error,error);
        if(error>(real?2e-10:0.0)) {
            std::cerr<<"short mismatch case="<<cases<<" n="<<n<<" bits="<<bits<<" d="<<dimensions<<" L="<<cutoff
                     <<" owner="<<i/bits<<" bit="<<i%bits<<" got="<<got[i]<<" want="<<expected[i]<<" error="<<error<<"\nQ:";
            for(auto x:q)std::cerr<<' '<<x;
            std::cerr<<"\nK:";
            for(auto x:k)std::cerr<<' '<<x;
            std::cerr<<'\n';
            std::exit(1);
        }
    }
    ++cases;values+=got.size();
}

static void test_dominance(std::mt19937_64& rng) {
    using short_detail::Event;
    for(int repeat=0;repeat<140;++repeat) {
        const int n=1+int(rng()%100),width=repeat%17==0?65:1+int(rng()%7),dimension=3+repeat%2;
        const int pcount=1+int(rng()%160),qcount=1+int(rng()%140);
        std::vector<double> weights(std::size_t(n)*width);
        for(double& w:weights)w=int(rng()%9)-4;
        std::vector<Event> p(pcount),q(qcount);
        for(Event& e:p) {
            for(int d=0;d<dimension;++d)e.coordinate[d]=int(rng()%n);
            e.id=int(rng()%n);e.sign=rng()%2?1:-1;e.query=false;
        }
        for(int i=0;i<qcount;++i) {
            Event& e=q[i];for(int d=0;d<dimension;++d)e.coordinate[d]=int(rng()%(n+2))-1;
            e.id=i;e.sign=rng()%2?1:-1;e.query=true;
        }
        std::vector<double> got(std::size_t(qcount)*width),expected(got.size());
        short_detail::Dominance engine(n,width,weights);engine.add(p,q,dimension,got);
        for(const Event& query:q)for(const Event& point:p) {
            bool accepted=true;for(int d=0;d<dimension;++d)accepted&=point.coordinate[d]<=query.coordinate[d];
            if(accepted)for(int c=0;c<width;++c)expected[std::size_t(query.id)*width+c]+=point.sign*query.sign*weights[std::size_t(point.id)*width+c];
        }
        if(got!=expected){std::cerr<<"dominance mismatch "<<repeat<<'\n';std::exit(1);}
        dominance_checks+=got.size();
    }
}

static int literal_common(const short_detail::VirtualIndex& ix,const short_detail::Word& a,const short_detail::Word& b) {
    int length=0;while(length<std::min(a.length,b.length) && ix.letter(a,length)==ix.letter(b,length))++length;
    return length;
}

static void test_virtual(std::mt19937_64& rng) {
    using namespace short_detail;
    for(int rep=0;rep<100;++rep) {
        const int n=1+int(rng()%45),cutoff=1+int(rng()%n);
        std::vector<std::uint64_t> q(n),k(n);
        for(auto& x:q)x=rep%2?rng():rng()%3;
        for(auto& x:k)x=rep%2?rng():rng()%3;
        if(rep%3==0){std::fill(q.begin(),q.end(),UINT64_MAX);std::fill(k.begin(),k.end(),UINT64_MAX);}
        Index index(q.data(),k.data(),n);VirtualIndex ix(index);
        const int last_active=int(rng()%n);SupportedWords support(index,last_active);
        for(int owner=0;owner<n-1;++owner)for(int bit:{0,1,63}) {
            const auto symbol=k[owner]^(std::uint64_t(1)<<bit);int expected=-1;
            for(int u=owner+1;u<=last_active;++u)if(q[u]==symbol) {
                int length=0;
                while(u+1+length<n && owner+1+length<n && q[u+1+length]==k[owner+1+length])++length;
                expected=std::max(expected,length);
            }
            if(support.right_extension(symbol,owner)!=expected){std::cerr<<"right-context support mismatch\n";std::exit(1);}
        }
        for(int s=0;s<n;++s) {
            const int bit=rep%3==0?63:int(rng()%64),m=std::min(n-s,cutoff+1);
            std::vector<Word> words;for(int r=s;r<s+m;++r)words.push_back(ix.edited(s,bit,r,cutoff));
            std::vector<int> order(m);std::iota(order.begin(),order.end(),0);
            std::sort(order.begin(),order.end(),[&](int a,int b){const int c=ix.compare(words[a],words[b]);return c?c<0:a<b;});
            ix.prepare(words,order);
            for(int w=0;w<m;++w)for(int depth=0;depth<=words[w].length+1;++depth) {
                const auto literal=ix.interval(n,[&](int p)->const Word&{return ix.query_words[p];},words[w],depth);
                if(ix.query_interval(words[w],depth,w)!=literal){std::cerr<<"prepared Q prefix interval mismatch\n";std::exit(1);}
            }
            for(int t=0;t<n;++t)for(int depth=0;depth<=std::min(t+2,cutoff+2);++depth) {
                const auto literal=ix.interval(m,[&](int p)->const Word&{return words[order[p]];},ix.row(t),depth);
                if(ix.word_interval(t,depth)!=literal){std::cerr<<"prepared word prefix interval mismatch\n";std::exit(1);}
            }
            std::vector<int> ml(m,-1),mh(m,-1);CompactTrie trie;trie.build(ix,words,order.data(),m,cutoff,ml,mh);
            for(const Piece& piece:trie.pieces)for(int qr=piece.lo;qr<piece.hi;++qr)
                if(literal_common(ix,words[piece.word],ix.query_words[qr])>piece.ceiling){std::cerr<<"trie edge ceiling mismatch\n";std::exit(1);}
            ActiveDictionary active;active.reset(words,order.data(),m,s);
            for(int t=0;t<n;++t) {
                const Word row=ix.row(t);Route expected,full;
                for(int w=0;w<m;++w) {
                    const int c=literal_common(ix,words[w],row);
                    if(c!=ix.common(words[w],row)){std::cerr<<"virtual LCP mismatch\n";std::exit(1);}
                    if(c) {
                        const Route r{c,words[w].end};if(r>full)full=r;
                        if(words[w].end<t && r>expected)expected=r;
                    }
                }
                active.advance(t);const Route got=active.query(ix,words,order.data(),row);
                if(got!=expected){std::cerr<<"active cutoff mismatch\n";std::exit(1);}
                const int found=trie.find(ix.row_rank[t]);
                if(full.length>0 && full.length<=cutoff) {
                    if(found<0 || Route{ix.common(words[found],row),words[found].end}!=full){std::cerr<<"trie mismatch\n";std::exit(1);}
                } else if(found>=0){std::cerr<<"trie mask mismatch\n";std::exit(1);}
            }
        }
    }
}

static void test_near_budget() {
    const std::vector<Route> ordered{{0,-1},{1,0},{1,7},{2,1},{2,91},{1024,4095},{1<<27,(1<<28)-2}};
    std::uint64_t previous_key=0;
    for(std::size_t i=0;i<ordered.size();++i) {
        const auto key=near_detail::order_key(ordered[i]);
        if(near_detail::route_from_key(key)!=ordered[i] || (i&&key<=previous_key)) {
            std::cerr<<"packed route ordering mismatch\n";std::exit(1);
        }
        previous_key=key;
    }
    for(int bit=0;bit<64;++bit) {
        const std::uint64_t value=UINT64_C(1)<<bit;
        if(near_detail::one_bit_index(value)!=bit || near_detail::one_bit_index_portable(value)!=bit) {
            std::cerr<<"portable bit index mismatch\n";std::exit(1);
        }
    }
    const std::vector<std::uint64_t> q{0,1,0,1},k{1,0,1,0};
    Index index(q.data(),k.data(),int(q.size()));
    const std::vector<double> weights(8,1.),payload(8,2.),empty(2,-1.);
    const Scores scores{4,2,weights.data(),payload.data(),empty.data()};
    for(bool full:{false,true}) {
        std::vector<double> credit(4,17.);Stats stats;
        stats.suffix_queries=123;stats.k_centers=456;stats.peak_workspace_bytes=789;
        if(near_k(index,scores,1,credit,stats,0,full) || credit!=std::vector<double>(4,17.)
            || stats.suffix_queries!=123 || stats.k_centers!=456 || stats.peak_workspace_bytes!=789) {
            std::cerr<<"near budget rejection changed output\n";std::exit(1);
        }
    }
}

int main(int argc,char** argv) {
    const bool quick=argc>1 && std::string(argv[1])=="--quick";
    const auto start=std::chrono::steady_clock::now();std::mt19937_64 rng(2026110608);
    test_dominance(rng);test_virtual(rng);test_near_budget();
    for(int n=1;n<=(quick?3:5);++n) {
        const std::uint64_t words=std::uint64_t(1)<<(2*n);
        for(std::uint64_t mask=0;mask<words;++mask) {
            std::vector<std::uint64_t> q(n),k(n);
            for(int t=0;t<n;++t){q[t]=(mask>>t)&1;k[t]=(mask>>(n+t))&1;}
            for(int cutoff=1;cutoff<=n;++cutoff)check(q,k,1,2,cutoff,rng);
        }
    }
    for(int rep=0;rep<(quick?50:500);++rep) {
        const int n=1+int(rng()%45),bits=1+int(rng()%4),d=1+int(rng()%7),cutoff=1+int(rng()%n);
        std::vector<std::uint64_t> q(n),k(n);
        for(int t=0;t<n;++t){q[t]=rng()%(1<<bits);k[t]=rng()%(1<<bits);}
        if(rep%5==0){for(int t=0;t<n;++t){q[t]=t%2;k[t]=0;}}
        if(rep%5==1){for(int t=0;t<n;++t){q[t]=t%3;k[t]=t%3;}k[rng()%n]^=std::uint64_t(1)<<(rng()%bits);}
        if(rep%5==2){std::fill(q.begin(),q.end(),0);std::fill(k.begin(),k.end(),0);k[rng()%n]=1;}
        check(q,k,bits,d,cutoff,rng,rep%7==0);
    }
    for(int rep=0;rep<(quick?8:40);++rep) {
        const int n=1+int(rng()%14),d=rep%5==0?65:1+int(rng()%7);
        std::vector<std::uint64_t> q(n),k(n);
        for(int t=0;t<n;++t){q[t]=rng()%2?UINT64_MAX:UINT64_MAX^(UINT64_C(1)<<63);k[t]=rng()%2?UINT64_MAX:UINT64_MAX^(UINT64_C(1)<<63);}
        check(q,k,64,d,1+int(rng()%n),rng,rep%7==0);
    }
    if(!quick)for(int n:{64,128,256})for(int cutoff:{1,4,16,n}) {
        std::vector<std::uint64_t> q(n),k(n,0);for(int t=0;t<n;++t)q[t]=t%2;
        check(q,k,2,3,cutoff,rng);
    }
    if(!quick)for(int n:{64,128})for(int repeat=0;repeat<4;++repeat) {
        const int bits=1+repeat%2;
        std::vector<std::uint64_t> q(n),k(n);
        for(int t=0;t<n;++t){q[t]=rng()%(1<<bits);k[t]=rng()%(1<<bits);}
        check(q,k,bits,3,16,rng,true);
    }
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout<<"{\"status\":\"PASS\",\"cases\":"<<cases<<",\"credit_elements\":"<<values
             <<",\"literal_route_checks\":"<<route_checks<<",\"dominance_components\":"<<dominance_checks
             <<",\"max_absolute_error\":"<<worst_error<<",\"seconds\":"<<seconds<<"}\n";
}
