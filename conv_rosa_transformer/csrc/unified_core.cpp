#include "approx_corridor_fft.h"
#include "exact_payload_identity_batch.h"
// Exact single-bit QKV-ROSA VJP, unified implementation.
// One algorithm: live Q repair states, deletion staircases, filtered K repair.
#include <algorithm>
#include <cstdint>
#include <exception>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>
#include "sam_q.h"
#include "forward.h"
#include "sam_routes.h"
#include "delete_indexed.h"
#include "exact_expansion.h"
#include "pair_candidates.h"
#include "full_word_map.h"

using PackedRoute = uint64_t;
static inline PackedRoute pack_route(int length, int end) {
    return length > 0 ? (uint64_t(uint32_t(length)) << 32) | uint32_t(end + 1) : 0;
}
static inline int route_end(PackedRoute r) { return int(uint32_t(r))-1; }

// Range additions / point snapshots cancel common Q deletion exactly on
// each detached time interval. Prior rows need not be installed: they cancel
// between a corridor's birth and finish snapshots.
struct QDeletionSnapshots {
    std::vector<rosa_exact::ExactExpansion> tree;
    int active=0,max_owner=-1;
    void init(int n) { if(tree.empty())tree.resize(n+2); }
    void add(int p,double v) { for(++p;p<int(tree.size());p+=p&-p)tree[p].add(v); }
    void range(int lo,int hi,double v) {
        // Every open snapshot's owner is <= max_owner. Other columns, and
        // rows crossed by no open snapshot, have no possible consumer.
        if(active && lo<=max_owner && v!=0.0) {
            hi=std::min(hi,max_owner);add(lo,v);add(hi+1,-v);
        }
    }
    rosa_exact::ExactExpansion point(int p) const {
        rosa_exact::ExactExpansion value;
        for(++p;p;p-=p&-p)value.add(tree[p]);
        return value;
    }
};
struct SingletonCorridor {
    int owner,bit,start,stop,length_bias,end_bias,next_finish;
    rosa_exact::ExactExpansion deletion_at_birth;
};
using CorridorHeapEntry=std::tuple<int,int,int>; // length bias, end bias, stop
struct CorridorHeap {
    size_t key;
    bool active=false;
    std::vector<CorridorHeapEntry> entries;
};

#if defined(_WIN32)
#define ROSA_EXPORT extern "C" __declspec(dllexport)
#else
#define ROSA_EXPORT extern "C"
#endif

// Both entry points use this route discovery and contraction. The optional
// quadratic model changes only the numerical score of an already fixed route.
// A compile-time flag leaves the ordinary VJP's arithmetic unchanged.
template<bool Quadratic>
static int uniform_impl(
    int n, int d, const uint64_t* q, const uint64_t* k,
    const uint8_t* v, const double* grad, const double* diagonal,
    const double* emb0, const double* emb1,
    double* cq, double* ck, double* cv, int32_t* lengths, int32_t* routes,
    uint64_t* counters, uint64_t repair_mask=~uint64_t(0),
    double* q_deletion=nullptr, double* k_deletion=nullptr,
    uint64_t* selection_counters=nullptr, int repair_budget=-1,
    uint64_t sampling_seed=0, uint64_t forced_mask=0,
    bool randomized_roots=false,
    uint64_t root_seed=rosa_exact::SamQRepairs::process_hash_seed(),
    uint64_t* spectral_stats=nullptr,int fft_sample_budget=0,uint64_t fft_sample_seed=0) {
  try {
    if(n<0 || n >= (1<<28) || d<1 || d>64) return 1;
    const size_t nd=size_t(n)*d;
    if(nd>size_t(std::numeric_limits<int>::max())) return 1;
    std::fill(cq,cq+nd,0.0); std::fill(ck,ck+nd,0.0); std::fill(cv,cv+nd,0.0);
    std::fill(lengths,lengths+n,0); std::fill(routes,routes+n,-1);
    std::fill(counters,counters+23,uint64_t(0));
    if(selection_counters)std::fill(selection_counters,selection_counters+5,uint64_t(0));
    if(!n) return 0;

    rosa_exact::FullWordMap payload_map(size_t(n),root_seed^UINT64_C(0xd6e8feb86659fd93));
    std::vector<uint64_t> payload_words;
    std::vector<int> payload_id(n);
    for(int s=0;s<n;++s) {
      uint64_t word=0;
      for(int j=0;j<d;++j)word|=uint64_t(v[size_t(s)*d+j])<<j;
      auto res=payload_map.insert(word,int(payload_words.size()));
      if(res.second)payload_words.push_back(word);
      payload_id[s]=res.first;
    }
    const int np=int(payload_words.size()); counters[0]=np;
    std::vector<double> payload,scores(np);
    if constexpr(!Quadratic) {
      payload.resize(size_t(np)*d);
      for(int p=0;p<np;++p) for(int j=0;j<d;++j)
        payload[size_t(p)*d+j]=((payload_words[p]>>j)&1) ? emb1[j] : -emb1[j];
    }
    // A diagonal output metric changes only row scores. There are just two
    // possible signed payload values per coordinate, so precompute their
    // baseline-relative costs once per row. This is algebraically equivalent
    // to scoring (g-h*Y)*P + h*P^2/2, but avoids subtracting huge common scores
    // when e0 and e1 are nearly equal. Baseline scores are exactly zero.
    // Ordinary contraction needs only this row. Allocate the two-cost history
    // only when a deferred corridor is born; no corridor can request earlier
    // rows than its own birth, so those earlier rows need never be retained.
    std::vector<double> bit_costs,row_bit_costs;
    if constexpr(Quadratic)row_bit_costs.resize(2*d);
    std::vector<int> score_stamp(np,-1);
    std::vector<rosa_exact::ExactExpansion> qdiff(n+1);
    std::vector<double> kdiff(n+1,0);
    std::vector<double> kgrad_diff;
    std::vector<PackedRoute> kbest;
    std::vector<size_t> ktouched;
    ktouched.reserve(n);

    rosa_exact::SamQRepairs qsam(q,k,n,d,repair_mask,repair_budget,sampling_seed,forced_mask,root_seed,randomized_roots);
    if(selection_counters) {
      selection_counters[0]=qsam.query_symbol_count();
      selection_counters[1]=qsam.root_repair_entries();
      selection_counters[2]=qsam.transition_working_bytes();
      selection_counters[3]=qsam.supported_repair_mask();
      selection_counters[4]=qsam.selected_repair_mask();
    }
    const bool has_repairs=qsam.selected_repair_mask()!=0;
    if(has_repairs)kbest.assign(nd,0);
    rosa_exact::SamRoutes index(qsam,q,k);
    rosa_exact::IndexedDeleteRow del(q,k,n);
    std::vector<int> qrun_end(has_repairs?n:0),krun_end(has_repairs?n:0),
        finish_head(n,-1),heap_at_key(has_repairs?nd:0,-1),active_heaps;
    for(int i=has_repairs?n-1:-1;i>=0;--i) {
      qrun_end[i]=(i+1<n && q[i]==q[i+1])?qrun_end[i+1]:i+1;
      krun_end[i]=(i+1<n && k[i]==k[i+1])?krun_end[i+1]:i+1;
    }
    std::vector<SingletonCorridor> corridors;
    rosa_random::BoundaryCorridorFFT corridor_fft(root_seed,fft_sample_budget,fft_sample_seed);
    std::vector<CorridorHeap> corridor_heaps;
    std::vector<rosa_exact::SamQRepairs::Repair> detached;
    // An identically zero output adjoint/metric contributes an exact zero to
    // every route score. Index only its nonzero support for deferred sums.
    // Removing add(0) events preserves the order of every nonzero event.
    std::vector<int> score_rows,score_row_end;
    std::vector<double> base_scores(has_repairs?n:0,0.0);
    // A single cache admission rule for all inputs. At most 4D payload
    // classes are retained, and row/class scores are evaluated on demand.
    const int cache_budget=4*d,cache_threshold=(n+cache_budget-1)/cache_budget;
    std::vector<int> payload_frequency(has_repairs?np:0,0),heavy_slot(has_repairs?np:0,-1);
    if(has_repairs)for(const int id:payload_id)++payload_frequency[id];
    int heavy_count=0;
    for(int id=0;has_repairs && id<np;++id)
      if(np<=cache_budget || payload_frequency[id]>=cache_threshold)heavy_slot[id]=heavy_count++;
    // Hoist the payload-position -> admitted slot map out of the H-row loop.
    // A cache hit then avoids a second dependent class-index lookup.
    std::vector<int> heavy_at_position(has_repairs?n:0);
    for(int s=0;has_repairs && s<n;++s)heavy_at_position[s]=heavy_slot[payload_id[s]];
    std::vector<double> corridor_cache;
    const double uncomputed=std::numeric_limits<double>::quiet_NaN();
    QDeletionSnapshots qdeletion;
    auto finish_corridors=[&](int t) {
      if(spectral_stats && finish_head[t]>=0)
        corridor_fft.finish<Quadratic>(finish_head[t],corridors,n,d,v,grad,bit_costs.data(),emb1,
          base_scores.data(),score_row_end.data(),heavy_at_position.data(),heavy_count);
      // The remaining numerical work visits each certified alignment interval.
      // Preserve all binary64 summation residuals until interval cancellation
      // is complete; the per-row dot product still uses ordinary float64.
      for(int id=finish_head[t];id>=0;id=corridors[id].next_finish) {
        auto&c=corridors[id];
        rosa_exact::ExactExpansion correction;
        double fft_value=0;
        if(spectral_stats && corridor_fft.get(id,t,fft_value))correction.add(fft_value);
        else for(int at=score_row_end[c.start];at<score_row_end[c.stop+1];++at) {
          const int x=score_rows[at];
          const int payload_position=x+c.end_bias+1,slot=heavy_at_position[payload_position];
          double row=slot<0?uncomputed:corridor_cache[size_t(x)*heavy_count+slot];
          if(std::isnan(row)) {
            const int payload_class=payload_id[payload_position];
            row=0;
            if constexpr(Quadratic) {
              const double* costs=bit_costs.data()+2*size_t(x)*d;
              const uint64_t word=payload_words[payload_class];
              for(int j=0;j<d;++j)row+=costs[2*j+int((word>>j)&1)];
            } else {
              const double* gx=grad+size_t(x)*d;
              const double* px=payload.data()+size_t(payload_class)*d;
              for(int j=0;j<d;++j)row+=gx[j]*px[j];
            }
            counters[21]+=d;
            if(slot>=0)corridor_cache[size_t(x)*heavy_count+slot]=row;
          }
          correction.add(row-base_scores[x]);
        }
        auto removed=qdeletion.point(c.owner);
        removed.subtract(c.deletion_at_birth);
        correction.subtract(removed);
        // This key never re-enters the ordinary live list. Reuse its finished
        // snapshot slot to retain the correction until the final common-Q sum
        // is available; projecting the two large terms separately loses bits.
        correction.add(cq[size_t(c.owner)*d+c.bit]);
        c.deletion_at_birth=std::move(correction);
        --qdeletion.active;
      }
    };
    for(int t=0;t<n;++t) {
      qsam.advance(t);
      index.observe();
      qsam.extract_singleton_continuations(k,detached);
      if(!detached.empty()) {
        qdeletion.init(n);
        if(score_row_end.empty()) {
          score_row_end.resize(n+1);score_rows.reserve(n);
          for(int x=0;x<n;++x) {
            score_row_end[x]=int(score_rows.size());
            bool nonzero=false;
            for(int j=0;j<d;++j) {
              const size_t at=size_t(x)*d+j;
              if(grad[at]!=0 || (Quadratic && diagonal[at]!=0)) {nonzero=true;break;}
            }
            if(nonzero)score_rows.push_back(x);
          }
          score_row_end[n]=int(score_rows.size());
        }
        if(heavy_count && corridor_cache.empty())corridor_cache.assign(size_t(n)*heavy_count,uncomputed);
        if constexpr(Quadratic)if(bit_costs.empty())bit_costs.resize(2*nd);
      }
      for(const auto&z:detached) {
        ++qdeletion.active;qdeletion.max_owner=std::max(qdeletion.max_owner,z.owner);
        int x=t+1,y=z.end+1;
        while(x<n && q[x]==k[y]) {
          const int step=std::min(qrun_end[x]-x,krun_end[y]-y);
          x+=step;y+=step;++counters[19];
        }
        const int stop=x-1;
        const int id=int(corridors.size());
        corridors.push_back({z.owner,z.bit,t,stop,z.length-t,z.end-t,finish_head[stop],qdeletion.point(z.owner)});
        finish_head[stop]=id;
        ++counters[17];counters[18]+=uint64_t(stop-t+1);
        const size_t key=size_t(z.end-t+z.owner)*d+z.bit;
        int hid=heap_at_key[key];
        if(hid<0) {
          hid=int(corridor_heaps.size());heap_at_key[key]=hid;
          corridor_heaps.push_back({key,false,{}});
        }
        auto&h=corridor_heaps[hid];
        h.entries.emplace_back(z.length-t,z.end-t,stop);
        std::push_heap(h.entries.begin(),h.entries.end());
        if(!h.active){h.active=true;active_heaps.push_back(hid);}
      }
      const auto base=qsam.baseline();
      const PackedRoute baseline=pack_route(base.length,base.end);
      lengths[t]=base.length; routes[t]=base.end;
      bool nonzero=false;
      for(int j=0;j<d;++j) {
        const size_t at=size_t(t)*d+j;
        if(grad[at]!=0 || (Quadratic && diagonal[at]!=0)){nonzero=true;break;}
      }
      if(!nonzero) {
        // Route/repair states above still advance. No deletion geometry or
        // row-local repair maxima are needed for an identically zero score.
        // Corridors ending here must still freeze their deletion snapshots.
        finish_corridors(t);
        continue;
      }
      del.reset(index,t,base.length,base.end);
      counters[10]+=del.a_record_count();
      counters[11]+=del.r_record_count();
      counters[13]+=del.a_expanded_record_count();
      const double* g=grad+size_t(t)*d;
      double empty_score=0;
      double* row_costs=nullptr;
      if constexpr(Quadratic) {
        const size_t offset=size_t(t)*d;
        const uint64_t baseline_word=base.end<0 ? 0 : payload_words[payload_id[base.end+1]];
        row_costs=bit_costs.empty() ? row_bit_costs.data() : bit_costs.data()+2*offset;
        for(int j=0;j<d;++j) {
          const double h=diagonal[offset+j];
          const double y=base.end<0 ? emb0[j] : ((baseline_word>>j)&1 ? emb1[j] : -emb1[j]);
          const double negative=-emb1[j]-y,positive=emb1[j]-y;
          row_costs[2*j]=g[j]*negative+(0.5*h*negative)*negative;
          row_costs[2*j+1]=g[j]*positive+(0.5*h*positive)*positive;
          const double empty=emb0[j]-y;
          empty_score+=g[j]*empty+(0.5*h*empty)*empty;
        }
      } else for(int j=0;j<d;++j)empty_score+=g[j]*emb0[j];
      if constexpr(Quadratic) {
        // The trust model is centered at this exact baseline payload. Its
        // score is zero, including when another endpoint shares the payload.
        // Populate the ordinary cache without summing D zero bit terms.
        if(base.end>=0) {
          const int id=payload_id[base.end+1];
          scores[id]=0.;score_stamp[id]=t;
          if(qdeletion.active && heavy_slot[id]>=0)
            corridor_cache[size_t(t)*heavy_count+heavy_slot[id]]=0.;
        }
      }
      auto score_end=[&](int r)->double {
        if(r<0)return empty_score;
        const int id=payload_id[r+1];
        if(score_stamp[id]!=t) {
          double value=0;
          if constexpr(Quadratic) {
            const uint64_t word=payload_words[id];
            for(int j=0;j<d;++j)value+=row_costs[2*j+int((word>>j)&1)];
          } else for(int j=0;j<d;++j)value+=g[j]*payload[size_t(id)*d+j];
          scores[id]=value;score_stamp[id]=t;counters[4]+=d;
          if(qdeletion.active && heavy_slot[id]>=0)
            corridor_cache[size_t(t)*heavy_count+heavy_slot[id]]=value;
        }
        return scores[id];
      };
      const double base_score=score_end(base.end);
      if(has_repairs)base_scores[t]=base_score;
      auto add_range=[&](std::vector<double>& diff,int lo,int hi,int r) {
        if(lo>hi)return;
        const double delta=score_end(r)-base_score;
        diff[lo]+=delta;diff[hi+1]-=delta;++counters[7];
      };
      for(const auto& z:del.qranges()) {
        const double delta=score_end(z.end)-base_score;
        qdiff[z.lo].add(delta);qdiff[z.hi+1].add(-delta);
        qdeletion.range(z.lo,z.hi,delta);++counters[7];
      }
      for(const auto& z:del.kranges())add_range(kdiff,z.lo,z.hi,z.end);
      // Sliding deletion endpoints end(s)=s-1 keep the owner payload.
      // Keep two baseline-bit gradient buckets; only the opposite bit enters
      // the final contraction. No subtraction of separately huge scores.
      if(!del.kowner_ranges().empty() && kgrad_diff.empty())
        kgrad_diff.assign(size_t(n+1)*d*2,0.0);
      for(const auto& z:del.kowner_ranges()) {
        double* left=kgrad_diff.data()+size_t(z.lo)*d*2;
        double* right=kgrad_diff.data()+size_t(z.hi+1)*d*2;
        const uint64_t base_word=payload_words[payload_id[base.end+1]];
        for(int j=0;j<d;++j) {
          const int bucket=2*j+int((base_word>>j)&1);
          const double value=Quadratic ? row_costs[2*j+int(!((base_word>>j)&1))] : g[j];
          left[bucket]+=value;right[bucket]-=value;
        }
        ++counters[14];counters[15]+=z.hi-z.lo+1;
      }

      // Ordinary live Q keys remain in the SAM; singleton continuations are
      // handled by the same repair decomposition through their interval events.
      auto qcursor=del.qcursor();
      for(const auto& z:qsam.repairs()) {
        const auto deletion=del.qroute_monotone(z.owner,qcursor);
        cq[size_t(z.owner)*d+z.bit]+=score_end(z.end)-score_end(deletion.end);
        ++counters[3];
      }

      auto consider=[&](int r,int owner_q,int bit,int repair_upper_length) {
        ++counters[1];
        const int a=t-owner_q, owner_k=r-a;
        const auto deletion=del.kroute(owner_k);
        if(pack_route(repair_upper_length,r)<=pack_route(deletion.length,deletion.end))return;
        const int b=owner_k ? index.lcs(owner_q-1,owner_k-1) : 0;
        const int length=a+1+b;
        ++counters[2];
        const PackedRoute candidate=pack_route(length,r);
        if(candidate<=pack_route(deletion.length,deletion.end))return;
        const size_t key=size_t(owner_k)*d+bit;
        if(!kbest[key])ktouched.push_back(key);
        kbest[key]=std::max(kbest[key],candidate);
      };

      // Any K repair also appears among the live Q repairs. An empty set is an
      // exact empty-work condition, not an input-family recognition heuristic.
      if(!qsam.repairs().empty()) {
        for(const auto& z:qsam.repairs()) {
          const int shortest=t-z.owner+1;
          const int short_state=z.short_state;
          const auto lower=del.min_k_delete_range(
              qsam.first_endpoint(short_state)-shortest+1,
              std::min(qsam.last_endpoint(short_state),t-1)-shortest+1);
          if(pack_route(z.length,z.end)<=pack_route(lower.length,lower.end))continue;
          const int required=std::max(lower.length,shortest);
          const int state=required==shortest ? short_state : qsam.state_for_length(z.state,required);
          qsam.for_each_causal_occurrence(state,[&](int r) {
            consider(r,z.owner,z.bit,z.length);
          });
        }
      }
      // A detached singleton has one K owner forever. Among all corridors
      // for the same owner/bit, route order depends only on fixed intercepts.
      size_t kept_heaps=0;
      for(const int hid:active_heaps) {
        auto&h=corridor_heaps[hid];
        while(!h.entries.empty() && std::get<2>(h.entries.front())<t) {
          std::pop_heap(h.entries.begin(),h.entries.end());h.entries.pop_back();++counters[22];
        }
        if(h.entries.empty()){h.active=false;continue;}
        active_heaps[kept_heaps++]=hid;++counters[20];
        const auto&top=h.entries.front();
        const PackedRoute candidate=pack_route(t+std::get<0>(top),t+std::get<1>(top));
        const auto deletion=del.kroute(int(h.key/d));
        if(candidate>pack_route(deletion.length,deletion.end)) {
          if(!kbest[h.key])ktouched.push_back(h.key);
          kbest[h.key]=std::max(kbest[h.key],candidate);
        }
      }
      active_heaps.resize(kept_heaps);
      for(size_t key:ktouched) {
        const auto deletion=del.kroute(int(key/d));
        ck[key]+=score_end(route_end(kbest[key]))-score_end(deletion.end);
        kbest[key]=0;
      }
      counters[3]+=ktouched.size();ktouched.clear();
      finish_corridors(t);
      if(baseline) {
        const size_t offset=size_t(base.end+1)*d;
        if constexpr(Quadratic) {
          const uint64_t word=payload_words[payload_id[base.end+1]];
          for(int j=0;j<d;++j)cv[offset+j]+=row_costs[2*j+int(!((word>>j)&1))];
        } else {
          const double* p=payload.data()+size_t(payload_id[base.end+1])*d;
          for(int j=0;j<d;++j)cv[offset+j]+=-2*g[j]*p[j];
        }
      }
    }
    counters[16]=index.materialized_index_flags();
    counters[5]=counters[1]; // Reporting now visits only causal endpoints.
    counters[6]=qsam.counters.active_advances;
    counters[8]=qsam.counters.transition_probes;
    counters[9]=qsam.counters.fallback_queries;
    counters[12]=qsam.counters.endpoint_queries;
    double ksum=0;
    std::vector<double> accumulated_kgrad(2*d,0.0);
    for(int s=0;s<n;++s) {
      // Replace consumed difference cells with their exact owner prefix.
      if(s)qdiff[s].add(qdiff[s-1]);
      ksum+=kdiff[s];
      const double common_q=qdiff[s].to_double();
      double owner_score=0;
      if(!kgrad_diff.empty()) {
        const uint64_t word=payload_words[payload_id[s]];
        const double* delta=kgrad_diff.data()+size_t(s)*d*2;
        for(int j=0;j<d;++j) {
          accumulated_kgrad[2*j]+=delta[2*j];
          accumulated_kgrad[2*j+1]+=delta[2*j+1];
          const bool bit=(word>>j)&1;
          // Only the opposite baseline bit can contribute. Matching bits
          // have an identically zero coefficient even at huge magnitudes.
          if constexpr(Quadratic)owner_score+=accumulated_kgrad[2*j+int(!bit)];
          else owner_score+=(bit?2:-2)*emb1[j]*accumulated_kgrad[2*j+int(!bit)];
        }
      }
      if(q_deletion)q_deletion[s]=common_q;
      if(k_deletion)k_deletion[s]=ksum+owner_score;
      for(int j=0;j<d;++j) {
        cq[size_t(s)*d+j]+=common_q;ck[size_t(s)*d+j]+=ksum+owner_score;
      }
    }
    rosa_exact::ExactPayloadIdentityBatch payload_identity;
    std::vector<rosa_exact::ExactPayloadIdentityBatch::Query> recovery_queries;
    uint64_t recovery_work=0;bool identity_built=false,identity_finite_known=false,identity_finite=true;
    if(spectral_stats) {
      for(size_t id=0;id<corridors.size();++id) {
        const auto&c=corridors[id];const bool sampled=spectral_stats&&corridor_fft.sampled(int(id),c.stop);
        const double threshold=corridor_fft.error_indicator(int(id),c.stop);
        auto total=c.deletion_at_birth;total.add(qdiff[c.owner]);
        if(sampled||(threshold>0&&std::abs(total.to_double())<=threshold)) {
          const int first=score_row_end[c.start],last=score_row_end[c.stop+1];
          bool possible=true;
          if(sampled)for(int at=first;at<std::min(last,first+8);++at) {
            const int x=score_rows[at],position=x+c.end_bias+1;
            if(routes[x]<0||payload_id[position]!=payload_id[routes[x]+1]){possible=false;break;}
          }
          if(!possible)continue;
          recovery_work+=last-first;
          if(first<last) {
            const int left=score_rows[first],right=score_rows[last-1];
            recovery_queries.push_back({int(id),left,left+c.end_bias+1,right-left+1});
          }
        }
      }
    }
    auto finite_identity_scores=[&]() {
      if(!identity_finite_known) {
        const auto&check=Quadratic?bit_costs:base_scores;
        identity_finite=std::all_of(check.begin(),check.end(),[](double x){return std::isfinite(x);});
        identity_finite_known=true;
      }
      return identity_finite;
    };
    int identity_levels=1;for(int z=spectral_stats?n:0;z>1;z>>=1)++identity_levels;
    if(recovery_work>8*uint64_t(n)*identity_levels&&finite_identity_scores()) {
      payload_identity.build(payload_id.data(),routes,n,recovery_queries,root_seed);identity_built=true;
    }
    size_t recovery_index=0;
    for(auto&c:corridors) {
      c.deletion_at_birth.add(qdiff[c.owner]);
      double value=c.deletion_at_birth.to_double(),raw=0;
      const int id=int(&c-corridors.data());
      const double threshold=spectral_stats?corridor_fft.error_indicator(id,c.stop):0.;
      const bool sampled=spectral_stats&&corridor_fft.sampled(id,c.stop);
      if((sampled||(threshold>0&&std::abs(value)<=threshold))&&corridor_fft.get(id,c.stop,raw)) {
        while(recovery_index<recovery_queries.size()&&recovery_queries[recovery_index].id<id)++recovery_index;
        bool identical=score_row_end[c.start]==score_row_end[c.stop+1];
        if(!identical&&identity_built)
          identical=recovery_index<recovery_queries.size()&&recovery_queries[recovery_index].id==id&&payload_identity.confirmed(int(recovery_index));
        if(!identical&&sampled&&!identity_built&&(!identity_finite_known||identity_finite)&&
            recovery_index<recovery_queries.size()&&recovery_queries[recovery_index].id==id) {
          identical=true;
          for(int at=score_row_end[c.start];at<score_row_end[c.stop+1];++at) {
            const int x=score_rows[at],position=x+c.end_bias+1;
            if(routes[x]<0||payload_id[position]!=payload_id[routes[x]+1]){identical=false;break;}
          }
          if(identical)identical=finite_identity_scores();
        }
        // Sampled nonzero estimates are never selected for exact recovery
        // by their random magnitude. Only input-fixed identities replace them.
        if(!sampled||identical) {
        rosa_exact::ExactExpansion exact;
        ++corridor_fft.counters.exact_fallbacks;
        if(!identical)for(int at=score_row_end[c.start];at<score_row_end[c.stop+1];++at) {
          const int x=score_rows[at];++corridor_fft.counters.fallback_rows;
          const int position=x+c.end_bias+1,slot=heavy_at_position[position];
          double row=slot<0?uncomputed:corridor_cache[size_t(x)*heavy_count+slot];
          if(std::isnan(row)) {
            const int payload_class=payload_id[position];row=0;
            if constexpr(Quadratic) {
              const double* costs=bit_costs.data()+2*size_t(x)*d;
              const uint64_t word=payload_words[payload_class];
              for(int j=0;j<d;++j)row+=costs[2*j+int((word>>j)&1)];
            } else {
              const double* gx=grad+size_t(x)*d;const double* px=payload.data()+size_t(payload_class)*d;
              for(int j=0;j<d;++j)row+=gx[j]*px[j];
            }
            counters[21]+=d;
            if(slot>=0)corridor_cache[size_t(x)*heavy_count+slot]=row;
          }
          exact.add(row-base_scores[x]);
        }
        c.deletion_at_birth.add(-raw);c.deletion_at_birth.add(exact);value=c.deletion_at_birth.to_double();
        }
      }
      cq[size_t(c.owner)*d+c.bit]=value;
    }
    // Causality makes these rows identically zero. Preserve that identity even
    // when range-difference prefix sums leave a roundoff cancellation residue.
    std::fill(cq,cq+d,0.0);
    std::fill(ck+size_t(n-1)*d,ck+nd,0.0);
    if(q_deletion)q_deletion[0]=0.0;
    if(k_deletion)k_deletion[n-1]=0.0;
    if(spectral_stats) {
      const auto& f=corridor_fft.counters;
      spectral_stats[0]=f.groups;spectral_stats[1]=f.queries;spectral_stats[2]=f.rows;
      spectral_stats[3]=f.transforms;spectral_stats[4]=f.columns;spectral_stats[5]=f.channels;
      spectral_stats[6]=f.exact_fallbacks;spectral_stats[7]=f.fallback_rows;
    }
    return 0;
  }catch(const std::bad_alloc&){return 2;}
   catch(const std::exception&){return 3;}
}

// Preserve the original C ABI and its first-order numerical path.
ROSA_EXPORT int rosa_uniform(
    int n, int d, const uint64_t* q, const uint64_t* k,
    const uint8_t* v, const double* grad, const double* emb0, const double* emb1,
    double* cq, double* ck, double* cv, int32_t* lengths, int32_t* routes,
    uint64_t* counters) {
  return uniform_impl<false>(n,d,q,k,v,grad,nullptr,emb0,emb1,
                             cq,ck,cv,lengths,routes,counters);
}

// Finite-flip credit of the supplied diagonal quadratic model. This does not
// estimate curvature and is not the Hessian of the hard threshold operation.
ROSA_EXPORT int rosa_quadratic(
    int n, int d, const uint64_t* q, const uint64_t* k,
    const uint8_t* v, const double* grad, const double* diagonal,
    const double* emb0, const double* emb1,
    double* cq, double* ck, double* cv, int32_t* lengths, int32_t* routes,
    uint64_t* counters) {
  return uniform_impl<true>(n,d,q,k,v,grad,diagonal,emb0,emb1,
                            cq,ck,cv,lengths,routes,counters);
}

// Compute every deletion baseline and V credit exactly, but discover repair
// alternatives only for selected complete bit lanes. A K owner/bit maximizes
// over multiple Q-side candidates, so sampling individual candidates instead
// would be biased. The returned unselected Q/K entries are their exact common
// deletion baseline; selected entries equal the full engine's finite credit.
ROSA_EXPORT int rosa_selected(
    int n, int d, uint64_t repair_mask,
    const uint64_t* q, const uint64_t* k,
    const uint8_t* v, const double* grad, const double* diagonal,
    const double* emb0, const double* emb1,
    double* cq, double* ck, double* cv, int32_t* lengths, int32_t* routes,
    uint64_t* counters, double* q_deletion, double* k_deletion,
    uint64_t* selection_counters) {
  if(diagonal)
    return uniform_impl<true>(n,d,q,k,v,grad,diagonal,emb0,emb1,
        cq,ck,cv,lengths,routes,counters,repair_mask,q_deletion,k_deletion,selection_counters);
  return uniform_impl<false>(n,d,q,k,v,grad,nullptr,emb0,emb1,
      cq,ck,cv,lengths,routes,counters,repair_mask,q_deletion,k_deletion,selection_counters);
}

// Support-certified repair sampling. All root joins are built once, exact
// impossible lanes are removed, then only complete selected repair lanes run.
// The numerical output is still unweighted selected credits plus baselines;
// the public wrapper applies known conditional inclusion probabilities.
ROSA_EXPORT int rosa_budgeted(
    int n, int d, int repair_budget, uint64_t sampling_seed, uint64_t forced_mask,
    const uint64_t* q, const uint64_t* k,
    const uint8_t* v, const double* grad, const double* diagonal,
    const double* emb0, const double* emb1,
    double* cq, double* ck, double* cv, int32_t* lengths, int32_t* routes,
    uint64_t* counters, double* q_deletion, double* k_deletion,
    uint64_t* selection_counters) {
  if(repair_budget<1 || repair_budget>d)return 1;
  if(diagonal)
    return uniform_impl<true>(n,d,q,k,v,grad,diagonal,emb0,emb1,
        cq,ck,cv,lengths,routes,counters,~uint64_t(0),q_deletion,k_deletion,
        selection_counters,repair_budget,sampling_seed,forced_mask);
  return uniform_impl<false>(n,d,q,k,v,grad,nullptr,emb0,emb1,
      cq,ck,cv,lengths,routes,counters,~uint64_t(0),q_deletion,k_deletion,
      selection_counters,repair_budget,sampling_seed,forced_mask);
}

// The forward uses the same SAM semantics with only baseline state.
// Ordinary suffix-link fallback is amortized linear; no repair indexes are built.
ROSA_EXPORT int rosa_matching_stats(
    int n, const uint64_t* q, const uint64_t* k,
    int32_t* lengths, int32_t* routes) {
  try {
    if (n<0 || n>=(1<<28)) return 1;
    rosa_exact::SamForward scan(q,k,n);
    scan.run(lengths,routes);
    return 0;
  } catch (const std::exception&) { return 2; }
    catch (...) { return 3; }
}

// The single native proposal path; only candidate selection is randomized.
ROSA_EXPORT int rosa_pair_candidates(
    int n,int d,const uint64_t*q,const uint64_t*k,const int32_t*rows,int row_count,
    const uint8_t*q_allowed,const uint8_t*k_allowed,int max_pairs,uint64_t seed,
    int bucket_cap,int occurrence_cap,int scan,int candidate_cap,
    uint64_t*pairs,uint64_t*trace,uint64_t*counts,double*timings) {
  try {
    if(n<0 || n>=(1<<28) || d<1 || d>64 || row_count<0 || row_count>n ||
       max_pairs<0 || bucket_cap<1 || occurrence_cap<1 || scan<1 || candidate_cap<max_pairs) return 1;
    if(!candidate_cap && max_pairs)return 1;
    return rosa_exact::propose_pair_candidates(n,d,q,k,rows,row_count,q_allowed,k_allowed,
        max_pairs,seed,bucket_cap,occurrence_cap,scan,candidate_cap,pairs,trace,counts,timings);
  }catch(const std::bad_alloc&){return 2;}catch(const std::exception&){return 3;}catch(...){return 4;}
}

// Batched ordinary-training API; reuses uniform_impl and the lean matcher.
#include "training_impl.h"
