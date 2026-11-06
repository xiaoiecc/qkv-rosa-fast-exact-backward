#pragma once
// Shared Q/K deletion and streamed owner/time projections. A deletion forbids
// a match from crossing one symbol; it never shifts the sequence.
// Arithmetic is reassociated binary64; route decisions are exact.
#include "index.h"
#include <array>
#include <cmath>
#include <queue>
#include <tuple>

namespace rosa_subq {
namespace delete_detail {

inline int square_root(int x,bool upward=false) {
    int a=int(std::sqrt(double(x)));
    while(std::int64_t(a+1)*(a+1)<=x)++a;
    while(std::int64_t(a)*a>x)--a;
    return upward && std::int64_t(a)*a<x?a+1:a;
}
inline void check_inputs(const Index& index,const Scores& scores,int width,const std::vector<int>* rows) {
    if(scores.n!=index.n || scores.d<1 || width<1 || !scores.weights || !scores.payload || !scores.empty)
        throw std::invalid_argument("invalid deletion dimensions");
    if(rows) {
        std::vector<unsigned char> used(index.n,0);
        for(int t:*rows) {
            if(t<0 || t>=index.n || used[t])throw std::invalid_argument("deletion rows must be distinct original indices");
            used[t]=1;
        }
    }
}
template<class Function>
inline void each_row(int n,const std::vector<int>* rows,Function&& fn) {
    if(rows){for(int t:*rows)fn(t);}else for(int t=0;t<n;++t)fn(t);
}

struct APPiece { int first=0,last=0,period=0,left=0,hq=0,fixed=0; };

// A constant-state generator of increasing endpoint progressions. Splitting
// the one equal-period-boundary exception needs at most three pending pieces.
class APScanner {
    const Index* index_=nullptr;
    Stats* stats_=nullptr;
    int t_=0,width_=1,next_start_=0,end_owner_=0,at_=0,size_=0;
    std::pair<int,int> ranks_;
    std::array<APPiece,3> pending_{};
public:
    APScanner()=default;
    APScanner(const Index& index,int t,int width,int start_owner,int end_owner,Stats& stats)
      :index_(&index),stats_(&stats),t_(t),width_(width),end_owner_(end_owner),
       ranks_(index.interval(t-width+1,width)) {
        next_start_=start_owner>=width-1?((start_owner-width+1)/width)*width:0;
    }
    bool next(APPiece& output) {
        for(;;) {
            if(at_<size_){output=pending_[at_++];++stats_->delete_pieces;return true;}
            if(!index_ || next_start_>t_-width_ || next_start_+width_>end_owner_)return false;
            const int lo=next_start_;next_start_+=width_;
            const int limit=std::min(t_,end_owner_)-width_;
            const int hi=std::min(lo+width_-1,limit);
            const int x=index_->position(ranks_,lo,hi,true);
            if(x<0)continue;
            const int y=index_->position(ranks_,x+1,hi,true);
            at_=size_=0;
            if(y<0) {
                const int r=x+width_-1;
                pending_[size_++]={r,r,0,0,0,index_->lcs(t_,r)};
                continue;
            }
            const int p=y-x;
            const auto& suffix=index_->forward_suffixes();const int offset=index_->n+1;
            const int right=std::min(index_->n-y,suffix.lcp_rank(suffix.rank[offset+x],suffix.rank[offset+y]));
            const int through=std::min(limit,y+right-width_);
            const int z=x+((through-x)/p)*p;
            // Two adjacent occurrences certify p-periodicity. Its exact
            // right extension may cross many discovery windows. No different
            // phase can occur inside it: shifting that occurrence back by a
            // multiple of p would place one strictly between x and y. The
            // original width-A window contains only this AP even if the run
            // ends before hi, so both portions can be skipped together.
            next_start_=std::max(hi,through)+1;
            const int a=x+width_-1,b=z+width_-1;
            const int hq=p+index_->same_lcs(true,t_,t_-p);
            const int hk=p+index_->same_lcs(false,b,b-p),left=b-hk+1;
            const int exception=left+hq-1;
            if(a<=exception && exception<=b && (exception-a)%p==0) {
                if(a<exception)pending_[size_++]={a,exception-p,p,left,hq,-1};
                pending_[size_++]={exception,exception,0,0,0,index_->lcs(t_,exception)};
                if(exception<b)pending_[size_++]={exception+p,b,p,left,hq,-1};
            } else pending_[size_++]={a,b,p,left,hq,-1};
        }
    }
};

struct RStep { int low,high,end; };
inline std::vector<RStep> row_steps(const Index& index,int t,int width,Stats& stats,std::size_t* workspace=nullptr) {
    const Route best=index.best(t);
    // R(t,h) can only move left as h increases. Equal endpoints at the two
    // limits certify one entire plateau, regardless of the underlying text.
    if(index.R(t,width)==best.end) {
        if(workspace)*workspace=std::max(*workspace,sizeof(RStep));
        return {{width,best.length,best.end}};
    }
    APScanner scanner(index,t,width,0,t,stats);APPiece piece;
    std::vector<Route> candidates;candidates.reserve(std::size_t(t/width+1)*2);
    while(scanner.next(piece)) {
        const int length=piece.period?std::min(piece.hq,piece.last-piece.left+1):piece.fixed;
        candidates.push_back({length,piece.last});
    }
    // AP windows and their split exceptions are already in endpoint order.
    // The Python prototype sorted this list; reversing it is sufficient.
    std::vector<RStep> steps;steps.reserve(candidates.size());int top=width-1;
    for(auto it=candidates.rbegin();it!=candidates.rend();++it)if(it->length>top) {
        steps.push_back({top+1,it->length,it->end});top=it->length;
    }
    if(workspace)*workspace=std::max(*workspace,candidates.capacity()*sizeof(Route)+steps.capacity()*sizeof(RStep));
    if(top!=best.length)throw std::logic_error("incomplete deletion occurrence cover");
    return steps;
}

struct RawRecord { int lo=0,first=0,last=0,period=0; };
struct OwnerSegment { int lo=0,hi=-1,period=0,residue=0,end=-1; };

// The prefix envelope A(s) is generated on demand. Each row retains one
// current record, at most one fixed tail, and a constant-state AP scanner.
class RowStream {
    int t_=0,cross_=0,end_=0,current_length_=0;
    APScanner scanner_;
    RawRecord current_;
    OwnerSegment tail_,lookahead_;
    bool has_current_=false,has_tail_=false,has_lookahead_=false;
    static int periodic_end(const OwnerSegment& segment,int owner) {
        int residue=(owner-segment.residue)%segment.period;
        if(residue<0)residue+=segment.period;
        return owner-1-residue;
    }
    static bool join(OwnerSegment& a,const OwnerSegment& b) {
        if(a.hi+1!=b.lo)return false;
        if(a.period && b.period) {
            if(a.period!=b.period || a.residue!=b.residue)return false;
        } else if(a.period) {
            if(periodic_end(a,b.lo)!=b.end || periodic_end(a,b.hi)!=b.end)return false;
        } else if(b.period) {
            if(periodic_end(b,a.lo)!=a.end || periodic_end(b,a.hi)!=a.end)return false;
            a.period=b.period;a.residue=b.residue;a.end=-1;
        } else if(a.end!=b.end)return false;
        a.hi=b.hi;return true;
    }
    bool next_record(RawRecord& output) {
        APPiece piece;
        while(scanner_.next(piece)) {
            int a=piece.first,b=piece.last;const int p=piece.period;
            if(b<cross_)continue;
            if(a>end_-1)return false;
            if(p) {
                if(a<cross_)a+=((cross_-a+p-1)/p)*p;
                if(b>end_-1)b-=((b-end_+1+p-1)/p)*p;
            } else if(a<cross_ || a>end_-1)continue;
            if(a>b)continue;
            if(!p) {
                if(piece.fixed<current_length_)continue;
                current_length_=piece.fixed;output={a+1,a,a,0};return true;
            }
            if(piece.hq<current_length_)continue;
            const int need=piece.left+current_length_-1;
            if(a<need)a+=((need-a+p-1)/p)*p;
            if(a>b)continue;
            current_length_=std::min(piece.hq,b-piece.left+1);
            output={a+1,a,b,p};return true;
        }
        return false;
    }
public:
    int row() const { return t_; }
    RowStream(const Index& index,int t,int width,int cross,int end,Stats& stats)
      :t_(t),cross_(cross),end_(end),
       scanner_(index,t,width,cross,end,stats) {
        const Route start=index.best(t,cross);current_length_=start.length;
        if(current_length_<width)throw std::logic_error("invalid deletion prefix crossing");
        current_={cross,start.end,start.end,0};has_current_=true;
    }
    bool next_piece(OwnerSegment& output) {
        if(has_tail_){output=tail_;has_tail_=false;return true;}
        while(has_current_) {
            const RawRecord record=current_;RawRecord after;
            const bool has_after=next_record(after);const int hi=has_after?after.lo-1:end_;
            has_current_=has_after;if(has_after)current_=after;
            if(record.lo>hi)continue;
            if(record.period && record.first<record.last) {
                const int end=std::min(hi,record.last+1);
                if(end<hi){tail_={std::max(record.lo,end+1),hi,0,0,record.last};has_tail_=tail_.lo<=tail_.hi;}
                if(record.lo<=end){output={record.lo,end,record.period,(record.first+1)%record.period,-1};return true;}
                if(has_tail_){output=tail_;has_tail_=false;return true;}
            } else {output={record.lo,hi,0,0,record.last};return true;}
        }
        return false;
    }
    bool next(OwnerSegment& output) {
        if(has_lookahead_){output=lookahead_;has_lookahead_=false;}
        else if(!next_piece(output))return false;
        OwnerSegment after;
        // AP windows are a discovery partition, not a required numerical
        // event boundary. Coalesce identical floor-affine routes across those
        // windows. A fixed tail is compatible exactly when both its endpoints
        // have the same floor value, since that function is monotone.
        while(next_piece(after)) {
            if(join(output,after))continue;
            lookahead_=after;has_lookahead_=true;break;
        }
        return true;
    }
};

struct PeriodState {
    bool made=false;
    int period=0,start=0,size=0,next_block=0;
    double carry=0;
    std::vector<double> coefficients,values,corrections;
};

// All residues of a period are contracted together. A new interval alters
// only one coefficient vector and at most two scalar correction positions.
// Period buffers are allocated only on their first actual event, so a stream
// with no periodic ranges does not pay the n*A numerical sweep.
class PeriodSweep {
    const Scores& scores_;
    std::vector<PeriodState> states_;
    std::vector<int> live_;
    static double dot(const double* a,const double* b,int d) {
        double result=0;for(int j=0;j<d;++j)result+=a[j]*b[j];return result;
    }
    void rebuild(PeriodState& state,int start) {
        const int p=state.period,d=scores_.d,n=scores_.n,m=std::min(p,n-start);
        state.start=start;state.size=m;state.next_block=start+p;state.carry=0;
        double base=0;
        for(int c=0;c<p;++c) {
            const double* a=state.coefficients.data()+std::size_t(c)*d;
            const double old=start?dot(a,scores_.payload+std::size_t(start-p+c)*d,d):0;
            base+=old;
            if(c<m)state.values[c]=dot(a,scores_.payload+std::size_t(start+c)*d,d)-old;
        }
        double prefix=0;
        for(int c=0;c<m;++c){prefix+=state.values[c];state.values[c]=base+prefix;}
        std::fill(state.corrections.begin(),state.corrections.begin()+m+1,0.0);
    }
public:
    PeriodSweep(const Scores& scores,int width):scores_(scores),states_(width) { live_.reserve(width); }
    void begin_owner(int owner) {
        for(int p:live_)if(states_[p].next_block==owner)rebuild(states_[p],owner);
    }
    void event(int owner,int p,int c,int row,int sign) {
        assert(p>0 && p<int(states_.size()) && c>=0 && c<p);
        auto& state=states_[p];const int d=scores_.d;
        if(!state.made) {
            state.made=true;state.period=p;state.coefficients.assign(std::size_t(p)*d,0.0);
            state.values.assign(p,0.0);state.corrections.assign(std::size_t(p)+1,0.0);
            state.start=(owner/p)*p;state.size=std::min(p,scores_.n-state.start);
            state.next_block=state.start+p;live_.push_back(p);
        }
        const int start=state.start,offset=owner-start;const double* weight=scores_.row(row);
        assert(offset>=0 && offset<state.size);
        if(offset<c) {
            const double old=start?dot(weight,scores_.payload+std::size_t(start-p+c)*d,d):0;
            state.corrections[offset]+=sign*old;
            if(c<state.size)state.corrections[c]+=sign*(dot(weight,scores_.payload+std::size_t(start+c)*d,d)-old);
        } else state.corrections[offset]+=sign*dot(weight,scores_.payload+std::size_t(start+c)*d,d);
        double* a=state.coefficients.data()+std::size_t(c)*d;
        for(int j=0;j<d;++j)a[j]+=sign*weight[j];
    }
    double finish_owner(int owner) {
        double result=0;
        for(int p:live_) {
            auto& state=states_[p];const int offset=owner-state.start;
            state.carry+=state.corrections[offset];result+=state.values[offset]+state.carry;
        }
        return result;
    }
    std::size_t workspace_bytes() const {
        std::size_t result=states_.capacity()*sizeof(PeriodState)+live_.capacity()*sizeof(int);
        for(int p:live_) {
            const auto& state=states_[p];
            result+=(state.coefficients.capacity()+state.values.capacity()+state.corrections.capacity())*sizeof(double);
        }
        return result;
    }
};

struct Event { int owner,sign,row,id,period,residue,hi; };
struct LaterEvent {
    bool operator()(const Event& a,const Event& b) const {
        return std::tie(a.owner,a.sign,a.row,a.period,a.residue,a.hi)
             > std::tie(b.owner,b.sign,b.row,b.period,b.residue,b.hi);
    }
};
class EventQueue:public std::priority_queue<Event,std::vector<Event>,LaterEvent> {
public:
    explicit EventQueue(std::size_t capacity){this->c.reserve(capacity);}
    std::size_t workspace_bytes() const{return this->c.capacity()*sizeof(Event);}
};

inline void fixed(std::vector<double>& differences,const Scores& scores,int lo,int hi,int t,int end) {
    if(lo>hi)return;
    assert(lo>=0 && hi<scores.n);
    const double value=scores.score(t,end);
    differences[lo]+=value;differences[hi+1]-=value;
}
} // namespace delete_detail

inline std::vector<double> q_deletion(const Index& index,const Scores& scores,int A,Stats& stats,
                                      const std::vector<int>* rows=nullptr) {
    using namespace delete_detail;check_inputs(index,scores,A,rows);A=std::min(A,index.n);
    std::vector<double> differences(std::size_t(index.n)+1,0.0),result(index.n,0.0);
    std::size_t row_workspace=0;
    each_row(index.n,rows,[&](int t) {
        const Route original=index.best(t);const int length=original.length;
        if(!length)return;
        const double base=scores.score(t,original.end);
        auto add=[&](int lo,int hi,int end) {
            if(lo>hi || end==original.end)return;
            const double value=scores.score(t,end)-base;
            differences[lo]+=value;differences[hi+1]-=value;
        };
        add(t,t,-1);
        if(length==1 || index.R(t,1)==original.end)return;
        for(int h=1;h<std::min(length+1,A);++h)add(t-h,t-h,index.R(t,h));
        if(length<A)return;
        for(const auto& step:row_steps(index,t,A,stats,&row_workspace))add(t-step.high,t-step.low,step.end);
    });
    double running=0;for(int s=0;s<index.n;++s){running+=differences[s];result[s]=running;}
    stats.peak_workspace_bytes=std::max<std::uint64_t>(stats.peak_workspace_bytes,index.workspace_bytes()
        +(differences.capacity()+result.capacity())*sizeof(double)+row_workspace);
    return result;
}

inline std::vector<double> k_deletion(const Index& index,const Scores& scores,int A,Stats& stats,
                                      const std::vector<int>* rows=nullptr) {
    using namespace delete_detail;check_inputs(index,scores,A,rows);const int n=index.n;A=std::min(A,n);
    std::vector<double> differences(std::size_t(n)+1,0.0),result(n,0.0);
    std::size_t row_workspace=0;
    std::vector<RowStream> streams;streams.reserve(rows?rows->size():std::size_t(n));
    EventQueue heap(rows?rows->size():std::size_t(n));
    auto pull=[&](int id,int owner_floor) {
        OwnerSegment segment;auto& stream=streams[id];const int t=stream.row();
        while(stream.next(segment)) {
            if(segment.lo<owner_floor)throw std::logic_error("deletion event regressed");
            if(!segment.period){fixed(differences,scores,segment.lo,segment.hi,t,segment.end);continue;}
            heap.push({segment.lo,1,t,id,segment.period,segment.residue,segment.hi});return;
        }
    };
    each_row(n,rows,[&](int t) {
        const Route original=index.best(t);if(!original.length)return;
        const int first=original.end-original.length+1,end=original.end-A;
        const double base=scores.score(t,original.end);
        differences[first]-=base;differences[original.end+1]+=base;
        if(first>end) {
            for(int s=first;s<=original.end;++s)fixed(differences,scores,s,s,t,index.deleted(t,s).end);
            return;
        }
        const auto steps=row_steps(index,t,A,stats,&row_workspace);
        int lo=first,hi=original.end+1;
        while(lo<hi) {
            const int s=lo+(hi-lo)/2;const Route left=index.best(t,s);
            const bool wins=left.length>=A && s+left.length>=index.R(t,left.length)+1;
            if(wins)hi=s;else lo=s+1;
        }
        const int cross=lo;
        // Once the prefix route wins with length >= A it keeps winning as
        // owner increases. Continue its AP stream through the short fringe;
        // only the portion before that exact crossing needs point queries.
        for(int s=std::max(first,end+1);s<std::min(original.end+1,cross);++s)
            fixed(differences,scores,s,s,t,index.deleted(t,s).end);
        for(std::size_t j=0;j<steps.size();++j) {
            const auto& step=steps[j];
            const int start=j+1==steps.size()?first:steps[j+1].end-step.high;
            fixed(differences,scores,std::max(first,start),std::min({end,cross-1,step.end-step.low}),t,step.end);
        }
        if(cross<=original.end) {
            const int id=int(streams.size());streams.emplace_back(index,t,A,cross,original.end,stats);pull(id,0);
        }
    });
    PeriodSweep sweep(scores,A);
    for(int owner=0;owner<n;++owner) {
        sweep.begin_owner(owner);
        while(!heap.empty() && heap.top().owner==owner) {
            const Event event=heap.top();heap.pop();++stats.delete_events;
            sweep.event(owner,event.period,event.residue,event.row,event.sign);
            if(event.sign>0) {
                if(event.hi+1<n)heap.push({event.hi+1,-1,event.row,event.id,event.period,event.residue,event.hi});
                else pull(event.id,n);
            } else pull(event.id,owner);
        }
        result[owner]=sweep.finish_owner(owner);
    }
    if(!heap.empty())throw std::logic_error("unconsumed deletion events");
    double running=0;for(int s=0;s<n;++s){running+=differences[s];result[s]+=running;}
    stats.peak_workspace_bytes=std::max<std::uint64_t>(stats.peak_workspace_bytes,index.workspace_bytes()
        +(differences.capacity()+result.capacity())*sizeof(double)+streams.capacity()*sizeof(RowStream)
        +heap.workspace_bytes()+std::max(row_workspace,sweep.workspace_bytes()));
    return result;
}

namespace delete_detail {
struct Snapshot { int block,owner,id,sign; };
template<bool QueryOwner>
inline std::vector<double> range_projections(const Index& index,const Scores& scores,
                                              const std::vector<OwnerRange>& queries,int H,Stats& stats) {
    check_inputs(index,scores,H,nullptr);const int n=index.n;H=std::min(H,n);
    if(queries.size()>std::size_t(std::numeric_limits<int>::max()/2))throw std::length_error("too many deletion range queries");
    std::vector<double> answer(queries.size(),0.0);if(queries.empty())return answer;
    std::vector<double> base(n);
    for(int t=0;t<n;++t)base[t]=scores.score(t,index.best(t).end);
    std::vector<Snapshot> events;events.reserve(queries.size()*2);
    auto point=[&](int owner,int t) {
        ++stats.range_fringes;
        if constexpr(QueryOwner) {
            const Route original=index.best(t);
            if(owner>t || t-owner>=original.length)return base[t];
            const int h=t-owner;return scores.score(t,h?index.R(t,h):-1);
        } else return scores.score(t,index.deleted(t,owner).end);
    };
    for(int j=0;j<int(queries.size());++j) {
        const auto& query=queries[j];
        if(query.owner<0 || query.owner>=n || query.lo<0 || query.lo>query.hi || query.hi>=n)
            throw std::invalid_argument("invalid deletion time range");
        const int first=(query.lo+H-1)/H,after=(query.hi+1)/H;
        if(first>=after)for(int t=query.lo;t<=query.hi;++t)answer[j]+=point(query.owner,t);
        else {
            events.push_back({first,query.owner,j,-1});events.push_back({after,query.owner,j,1});
            for(int t=query.lo;t<first*H;++t)answer[j]+=point(query.owner,t);
            for(int t=after*H;t<=query.hi;++t)answer[j]+=point(query.owner,t);
        }
    }
    if(events.empty())return answer;
    std::sort(events.begin(),events.end(),[](const Snapshot& a,const Snapshot& b){
        return std::tie(a.block,a.id,a.sign)<std::tie(b.block,b.id,b.sign);
    });
    std::vector<double> prefix(n,0.0);std::vector<int> rows(H);std::size_t at=0;
    const int A=std::max(1,square_root(QueryOwner?n:H,!QueryOwner));int active=0;
    const std::size_t outer_workspace=(base.capacity()+prefix.capacity()+answer.capacity())*sizeof(double)
        +events.capacity()*sizeof(Snapshot)+rows.capacity()*sizeof(int);
    for(int block=events.front().block;block<=events.back().block;++block) {
        while(at<events.size() && events[at].block==block) {
            const auto& event=events[at++];answer[event.id]+=event.sign*prefix[event.owner];active-=event.sign;
        }
        if(block==events.back().block)break;
        // Unrequested gaps cancel out of every future prefix difference.
        if(!active)continue;
        const int start=block*H;double baseline_sum=0;
        for(int z=0;z<H;++z){rows[z]=start+z;baseline_sum+=base[start+z];}
        std::vector<double> delta;
        const auto previous_peak=stats.peak_workspace_bytes;stats.peak_workspace_bytes=0;
        if constexpr(QueryOwner)delta=q_deletion(index,scores,A,stats,&rows);
        else delta=k_deletion(index,scores,A,stats,&rows);
        stats.peak_workspace_bytes=std::max<std::uint64_t>(previous_peak,stats.peak_workspace_bytes+outer_workspace);
        for(int owner=0;owner<n;++owner)prefix[owner]+=delta[owner]+baseline_sum;
        ++stats.range_blocks;
    }
    stats.peak_workspace_bytes=std::max<std::uint64_t>(stats.peak_workspace_bytes,index.workspace_bytes()+outer_workspace);
    return answer;
}
} // namespace delete_detail

inline std::vector<double> q_delete_ranges(const Index& index,const Scores& scores,
                                            const std::vector<OwnerRange>& queries,int H,Stats& stats) {
    return delete_detail::range_projections<true>(index,scores,queries,H,stats);
}
inline std::vector<double> k_delete_ranges(const Index& index,const Scores& scores,
                                            const std::vector<OwnerRange>& queries,int H,Stats& stats) {
    return delete_detail::range_projections<false>(index,scores,queries,H,stats);
}
} // namespace rosa_subq
