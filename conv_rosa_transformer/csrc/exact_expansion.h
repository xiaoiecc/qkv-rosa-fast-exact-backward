#pragma once
// Exact sums of finite binary64 values under IEEE round-to-nearest,
// gradual underflow, and no intermediate overflow. Do not use fast-math.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>
namespace rosa_exact {
class ExactExpansion {
    double lo_=0.0, hi_=0.0;
    std::vector<double>* spill_=nullptr; // present only with >2 components
    static void two_sum(double a,double b,double& sum,double& error) {
        sum=a+b;
        const double bv=sum-a;
        error=(a-(sum-bv))+(b-bv);
    }
    // Normalize two exact components with FastTwoSum after ordering them.
    // This preserves their exact sum, including cancellation and subnormals.
    void normalize_pair(double a,double b) {
        if(std::abs(a)<std::abs(b))std::swap(a,b);
        const double sum=a+b;
        lo_=b-(sum-a);hi_=sum;
    }
    void finish_grow(double error0,double error1,double high) {
        double terms[3];int count=0;
        if(error0!=0.0)terms[count++]=error0;
        if(error1!=0.0)terms[count++]=error1;
        if(high!=0.0)terms[count++]=high;
        if(count<=2) {
            lo_=count==2?terms[0]:0.0;hi_=count?terms[count-1]:0.0;
            return;
        }
        // Three grow outputs need not require three stored components.
        // If the low pair adds exactly, normalize that pair with the high
        // component, retaining two components without ever allocating.
        double joined,tail;two_sum(terms[0],terms[1],joined,tail);
        if(tail==0.0){normalize_pair(terms[2],joined);return;}
        spill_=new std::vector<double>();spill_->reserve(4);
        spill_->assign(terms,terms+3);
    }
    void add_spilled(double value) {
        auto& e=*spill_;
        const std::size_t old=e.size();
        std::size_t write=0;
        double q=value;
        for(std::size_t i=0;i<old;++i) {
            double sum,error;two_sum(q,e[i],sum,error);q=sum;
            if(error!=0.0)e[write++]=error;
        }
        if(q!=0.0) {
            if(write<old)e[write]=q;else e.push_back(q);
            ++write;
        }
        e.resize(write);
        if(write==3) {
            double joined,tail;two_sum(e[0],e[1],joined,tail);
            if(tail==0.0) {
                normalize_pair(e[2],joined);
                delete spill_;spill_=nullptr;return;
            }
        }
        if(write<=2) {
            lo_=write==2 ? e[0] : 0.0;
            hi_=write ? e[write-1] : 0.0;
            delete spill_;spill_=nullptr;
        }
    }
    static bool even(double value) {
        uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));return !(bits&1);
    }
public:
    ExactExpansion()=default;
    explicit ExactExpansion(double value):hi_(value) {}
    ~ExactExpansion(){delete spill_;}
    ExactExpansion(const ExactExpansion& x):lo_(x.lo_),hi_(x.hi_),spill_(x.spill_?new std::vector<double>(*x.spill_):nullptr) {}
    ExactExpansion(ExactExpansion&& x) noexcept:lo_(x.lo_),hi_(x.hi_),spill_(x.spill_){x.lo_=x.hi_=0.0;x.spill_=nullptr;}
    ExactExpansion& operator=(ExactExpansion x) noexcept {swap(x);return *this;}
    void swap(ExactExpansion& x) noexcept {std::swap(lo_,x.lo_);std::swap(hi_,x.hi_);std::swap(spill_,x.spill_);}
    void clear(){lo_=hi_=0.0;delete spill_;spill_=nullptr;}
    void add(double value) {
        if(value==0.0)return;
        if(spill_){add_spilled(value);return;}
        // Always perform the same normalized inline update. Zero input,
        // zero low component, and zero first residual use these identical
        // error-free operations; only a genuine remaining tail needs grow.
        double sum,error;two_sum(hi_,value,sum,error);
        double low,tail;two_sum(lo_,error,low,tail);
        normalize_pair(sum,low);
        if(tail==0.0)return;
        // The exact update is hi_+lo_+tail. Reinsert the remaining tail
        // through an error-free grow; its zero-eliminated output is again
        // ordered/nonoverlapping, including unusually wide exponents.
        double q,error0,high,error1;
        two_sum(tail,lo_,q,error0);
        two_sum(q,hi_,high,error1);
        finish_grow(error0,error1,high);
    }
    template<class Fn> void for_each_component(Fn&& fn) const {
        if(spill_){for(double value:*spill_)fn(value);}
        else {if(lo_!=0.0)fn(lo_);if(hi_!=0.0)fn(hi_);}
    }
    void add(const ExactExpansion& x) {
        if(this==&x){ExactExpansion copy(x);add(copy);return;}
        x.for_each_component([&](double value){add(value);});
    }
    void subtract(const ExactExpansion& x) {
        if(this==&x){clear();return;}
        x.for_each_component([&](double value){add(-value);});
    }
    std::size_t component_count() const {return spill_?spill_->size():std::size_t(lo_!=0.0)+std::size_t(hi_!=0.0);}
    bool spilled() const {return spill_!=nullptr;}
    int sign() const {const double x=spill_?spill_->back():hi_;return (x>0.0)-(x<0.0);}
    // Projection occurs only after all range-query and snapshot subtraction.
    // Two components require one ordinary, correctly rounded addition.
    double to_double() const {
        if(!spill_)return hi_+lo_;
        double candidate=0.0;for(double value:*spill_)candidate+=value;
        // Compare the exact residual with an exact adjacent-float midpoint.
        // This handles the rare multi-component tie without throwing away
        // lower components before deciding round-to-nearest, ties-to-even.
        for(;;) {
            ExactExpansion residual(*this);residual.add(-candidate);
            const int direction=residual.sign();if(!direction)return candidate;
            const double neighbor=std::nextafter(candidate,direction>0?std::numeric_limits<double>::infinity():-std::numeric_limits<double>::infinity());
            if(!std::isfinite(neighbor))return candidate; // no-overflow contract
            const double half=std::abs(neighbor-candidate)*0.5;
            if(half==0.0){candidate=neighbor;continue;}
            residual.add(direction>0?-half:half);
            const int comparison=direction*residual.sign();
            if(comparison<0 || (comparison==0 && even(candidate)))return candidate;
            if(comparison==0)return neighbor;
            candidate=neighbor;
        }
    }
};
}
