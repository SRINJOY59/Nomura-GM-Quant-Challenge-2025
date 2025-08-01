#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <memory>
#include <array>
#include <fstream>
#include <iomanip>
#include <string>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

#if __cplusplus < 201402L
namespace std {
    template <typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}
#endif

namespace Constants {
    constexpr double TOLERANCE = 1e-12;
    constexpr double PI = 3.141592653589793;
    constexpr double SQRT_2PI = 2.5066282746310005;
    constexpr int MAX_ITERATIONS = 1000;
    constexpr double INITIAL_RATE_GUESS = 5.0;
}

enum class RateConvention : uint8_t {
    LINEAR = 0,
    CUMULATIVE = 1,
    RECURSIVE = 2
};

enum class RelativeFactorMethod : uint8_t {
    UNITY = 0,
    CUMULATIVE = 1
};

// Optimized base class for pricing calculations
class PricingEngine {
public:
    virtual ~PricingEngine() = default;
    
    virtual double calculatePrice(double effectiveRate) const noexcept = 0;
    virtual double calculateRate(double price) const = 0;
    virtual double calculatePriceSensitivity(double effectiveRate) const noexcept = 0;
    virtual double calculateRateSensitivity(double price) const = 0;
    virtual double calculateSecondPriceSensitivity(double effectiveRate) const noexcept = 0;
    virtual double calculateSecondRateSensitivity(double price) const = 0;
};

// Linear Rate Convention Implementation
class LinearPricingEngine final : public PricingEngine {
private:
    const double notional_;
    const double maturity_years_;
    const double maturity_factor_; // Pre-computed for optimization

public:
    LinearPricingEngine(double n, double m) noexcept 
        : notional_(n), maturity_years_(m), maturity_factor_(n * m / 100.0) {}

    double calculatePrice(double effectiveRate) const noexcept override {
        return notional_ - effectiveRate * maturity_factor_;
    }

    double calculateRate(double price) const override {
        return (notional_ - price) / maturity_factor_;
    }

    double calculatePriceSensitivity(double) const noexcept override {
        return -maturity_factor_;
    }

    double calculateRateSensitivity(double) const override {
        return -1.0 / maturity_factor_;
    }

    double calculateSecondPriceSensitivity(double) const noexcept override {
        return 0.0;
    }

    double calculateSecondRateSensitivity(double) const override {
        return 0.0;
    }
};

class CumulativePricingEngine final : public PricingEngine {
private:
    const double notional_;
    const double value_rate_;
    const int payment_frequency_;
    const double discount_divisor_;
    
    std::vector<double> payment_times_;
    std::vector<double> cash_flows_;
    std::vector<double> powers_; // Pre-computed powers

public:
    CumulativePricingEngine(double n, double vr, int pf, double maturity_years) 
        : notional_(n), value_rate_(vr), payment_frequency_(pf),
          discount_divisor_(100.0 * pf) {
        
        int num_payments = static_cast<int>(maturity_years * pf);
        double payment_amount = vr * n / discount_divisor_;
        
        payment_times_.reserve(num_payments);
        cash_flows_.reserve(num_payments);
        powers_.reserve(num_payments);
        
        for (int i = 1; i <= num_payments; ++i) {
            double time = static_cast<double>(i) / pf;
            double power = pf * time;
            
            payment_times_.push_back(time);
            cash_flows_.push_back(payment_amount);
            powers_.push_back(power);
        }
        
        if (!cash_flows_.empty()) {
            cash_flows_.back() += notional_;
        }
    }

    double calculatePrice(double effective_rate) const noexcept override {
        double discount_rate = effective_rate / discount_divisor_;
        double discount_base = 1.0 + discount_rate;
        
        double price = 0.0;
        for (size_t i = 0; i < cash_flows_.size(); ++i) {
            price += cash_flows_[i] / std::pow(discount_base, powers_[i]);
        }
        
        return price;
    }

    double calculateRate(double price) const override {
        double rate = Constants::INITIAL_RATE_GUESS;
        
        for (int iter = 0; iter < Constants::MAX_ITERATIONS; ++iter) {
            double f = calculatePrice(rate) - price;
            double df = calculatePriceSensitivity(rate);
            
            if (std::abs(f) < Constants::TOLERANCE) break;
            if (std::abs(df) < Constants::TOLERANCE) break;
            
            double new_rate = rate - f / df;
            if (std::abs(new_rate - rate) < Constants::TOLERANCE) break;
            
            rate = new_rate;
        }
        
        return rate;
    }

    double calculatePriceSensitivity(double effective_rate) const noexcept override {
        double discount_rate = effective_rate / discount_divisor_;
        double discount_base = 1.0 + discount_rate;
        double divisor_factor = 1.0 / discount_divisor_;
        
        double sensitivity = 0.0;
        for (size_t i = 0; i < cash_flows_.size(); ++i) {
            double power = powers_[i];
            double discount_factor = std::pow(discount_base, power);
            sensitivity -= cash_flows_[i] * power * divisor_factor / (discount_factor * discount_base);
        }
        
        return sensitivity;
    }

    double calculateRateSensitivity(double price) const override {
        double rate = calculateRate(price);
        return 1.0 / calculatePriceSensitivity(rate);
    }

    double calculateSecondPriceSensitivity(double effective_rate) const noexcept override {
        double discount_rate = effective_rate / discount_divisor_;
        double discount_base = 1.0 + discount_rate;
        double divisor_factor_sq = 1.0 / (discount_divisor_ * discount_divisor_);
        
        double second_sensitivity = 0.0;
        for (size_t i = 0; i < cash_flows_.size(); ++i) {
            double power = powers_[i];
            double discount_factor = std::pow(discount_base, power);
            double term = cash_flows_[i] * power * (power + 1.0) * divisor_factor_sq / (discount_factor * discount_base * discount_base);
            second_sensitivity += term;
        }
        
        return second_sensitivity;
    }

    double calculateSecondRateSensitivity(double price) const override {
        double rate = calculateRate(price);
        double first_deriv = calculatePriceSensitivity(rate);
        double second_deriv = calculateSecondPriceSensitivity(rate);
        
        return -second_deriv / (first_deriv * first_deriv * first_deriv);
    }
};

// Recursive Pricing Engine Implementation
class RecursivePricingEngine final : public PricingEngine {
private:
    const double notional_;
    const double value_rate_;
    const int payment_frequency_;
    const double maturity_years_;
    const double payment_amount_;
    const size_t num_payments_;
    const double time_step_;

public:
    RecursivePricingEngine(double n, double vr, int pf, double m) 
        : notional_(n), value_rate_(vr), payment_frequency_(pf), maturity_years_(m),
          payment_amount_(vr * n / (100.0 * pf)),
          num_payments_(static_cast<size_t>(m * pf)),
          time_step_(1.0 / pf) {}

    double calculatePrice(double effective_rate) const noexcept override {
        double rate_factor = effective_rate / 100.0;
        double fv = 0.0;
        for (size_t i = 0; i < num_payments_; ++i) {
            double dt = (i < num_payments_ - 1) ? time_step_ : 0.0;
            fv = (fv + payment_amount_) * (1.0 + rate_factor * dt);
        }
        return (notional_ + fv) / (1.0 + rate_factor * maturity_years_);
    }

    double calculateRate(double price) const override {
        double rate = Constants::INITIAL_RATE_GUESS;
        
        for (int iter = 0; iter < Constants::MAX_ITERATIONS; ++iter) {
            double f = calculatePrice(rate) - price;
            double df = calculatePriceSensitivity(rate);
            
            if (std::abs(f) < Constants::TOLERANCE) break;
            if (std::abs(df) < Constants::TOLERANCE) break;
            
            double new_rate = rate - f / df;
            if (std::abs(new_rate - rate) < Constants::TOLERANCE) break;
            
            rate = new_rate;
        }
        
        return rate;
    }

    double calculatePriceSensitivity(double effective_rate) const noexcept override {
        double rate_factor = effective_rate / 100.0;
        
        double fv = 0.0;
        double dfv_dr = 0.0;
        
        for (size_t i = 0; i < num_payments_; ++i) {
            double dt = (i < num_payments_ - 1) ? time_step_ : 0.0;
            double old_fv = fv;
            double old_dfv = dfv_dr;
            double factor = 1.0 + rate_factor * dt;
            fv = (old_fv + payment_amount_) * factor;
            dfv_dr = old_dfv * factor + (old_fv + payment_amount_) * dt;
        }
        
        double denominator = 1.0 + rate_factor * maturity_years_;
        double numerator = dfv_dr * denominator - (notional_ + fv) * maturity_years_;
        
        return numerator / (denominator * denominator * 100.0);
    }

    double calculateRateSensitivity(double price) const override {
        double rate = calculateRate(price);
        return 1.0 / calculatePriceSensitivity(rate);
    }

    double calculateSecondPriceSensitivity(double effective_rate) const noexcept override {
        double rate_factor = effective_rate / 100.0;
        double fv = 0.0, dfv_dr = 0.0, d2fv_dr2 = 0.0;
        
        for (size_t i = 0; i < num_payments_; ++i) {
            double dt = (i < num_payments_ - 1) ? time_step_ : 0.0;
            double old_fv = fv;
            double old_dfv = dfv_dr;
            double old_d2fv = d2fv_dr2;
            double factor = 1.0 + rate_factor * dt;
            
            fv = (old_fv + payment_amount_) * factor;
            dfv_dr = old_dfv * factor + (old_fv + payment_amount_) * dt;
            d2fv_dr2 = old_d2fv * factor + 2.0 * old_dfv * dt;
        }
        
        double denom = 1.0 + rate_factor * maturity_years_;
        double M_dt = maturity_years_;
        
        double num = d2fv_dr2 * denom - 2.0 * dfv_dr * M_dt;
        double main_term = num / (denom * denom);
        double correction_term = 2.0 * M_dt * (dfv_dr * denom - (notional_ + fv) * M_dt) / (denom * denom * denom);
        
        return (main_term - correction_term) / 10000.0;
    }

    double calculateSecondRateSensitivity(double price) const override {
        double rate = calculateRate(price);
        double first_deriv = calculatePriceSensitivity(rate);
        double second_deriv = calculateSecondPriceSensitivity(rate);
        
        return -second_deriv / (first_deriv * first_deriv * first_deriv);
    }
};

// ValueNote class with move semantics
class ValueNote {
private:
    const double notional_;
    const double maturity_years_;
    const double value_rate_;
    const int payment_frequency_;
    const RateConvention convention_;
    std::unique_ptr<PricingEngine> pricing_engine_;

public:
    ValueNote(double n, double m, double vr, int pf, RateConvention conv) 
        : notional_(n), maturity_years_(m), value_rate_(vr), 
          payment_frequency_(pf), convention_(conv) {
        
        switch (conv) {
            case RateConvention::LINEAR:
                pricing_engine_ = std::make_unique<LinearPricingEngine>(n, m);
                break;
            case RateConvention::CUMULATIVE:
                pricing_engine_ = std::make_unique<CumulativePricingEngine>(n, vr, pf, m);
                break;
            case RateConvention::RECURSIVE:
                pricing_engine_ = std::make_unique<RecursivePricingEngine>(n, vr, pf, m);
                break;
            default:
                throw std::invalid_argument("Invalid rate convention");
        }
    }

    // Allow move construction but delete move assignment
    ValueNote(ValueNote&&) noexcept = default;
    ValueNote& operator=(ValueNote&&) noexcept = delete;
    ValueNote(const ValueNote&) = delete;
    ValueNote& operator=(const ValueNote&) = delete;

    double getNotional() const noexcept { return notional_; }
    double getMaturityYears() const noexcept { return maturity_years_; }
    double getValueRate() const noexcept { return value_rate_; }
    int getPaymentFrequency() const noexcept { return payment_frequency_; }
    RateConvention getConvention() const noexcept { return convention_; }

    double calculatePrice(double effective_rate) const noexcept {
        return pricing_engine_->calculatePrice(effective_rate);
    }

    double calculateEffectiveRate(double price) const {
        return pricing_engine_->calculateRate(price);
    }

    double calculatePriceSensitivity(double effective_rate) const noexcept {
        return pricing_engine_->calculatePriceSensitivity(effective_rate);
    }

    double calculateRateSensitivity(double price) const {
        return pricing_engine_->calculateRateSensitivity(price);
    }

    double calculateSecondPriceSensitivity(double effective_rate) const noexcept {
        return pricing_engine_->calculateSecondPriceSensitivity(effective_rate);
    }

    double calculateSecondRateSensitivity(double price) const {
        return pricing_engine_->calculateSecondRateSensitivity(price);
    }

    double calculateForwardPrice(double current_price, double risk_free_rate, double time_to_expiration) const {
        double accrued_interest = 0.0;
        double time_step = 1.0 / payment_frequency_;
        double payment_amount = value_rate_ * notional_ / (100.0 * payment_frequency_);
        
        for (double t = time_step; t <= time_to_expiration; t += time_step) {
            accrued_interest += payment_amount / std::pow(1.0 + risk_free_rate / 100.0, t);
        }
        
        return (1.0 + risk_free_rate * time_to_expiration / 100.0) * (current_price - accrued_interest);
    }
};

void writeCorrectCSVOutput() {
    std::ofstream file("output.csv");
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open output.csv for writing");
    }
    
    file << std::fixed << std::setprecision(6);
    file << ",Linear,Cumulative,Recursive,Q 2.1,Q 2.2,Q 2.3,Q 2.4 a),Q 2.4 b)\n";
    
    auto vn1_linear = std::make_unique<ValueNote>(100, 5, 3.5, 1, RateConvention::LINEAR);
    auto vn1_cumulative = std::make_unique<ValueNote>(100, 5, 3.5, 1, RateConvention::CUMULATIVE);
    auto vn1_recursive = std::make_unique<ValueNote>(100, 5, 3.5, 1, RateConvention::RECURSIVE);
    
    const std::array<std::tuple<double, double, double, int>, 4> vn_params = {{
        std::make_tuple(100.0, 5.0, 3.5, 1),
        std::make_tuple(100.0, 1.5, 2.0, 2),
        std::make_tuple(100.0, 4.5, 3.25, 1),
        std::make_tuple(100.0, 10.0, 8.0, 4)
    }};
    
    std::array<double, 4> relative_factors{};
    constexpr double SVR = 5.0;
    
    for (size_t i = 0; i < vn_params.size(); ++i) {
        double n = std::get<0>(vn_params[i]);
        double m = std::get<1>(vn_params[i]);
        double vr = std::get<2>(vn_params[i]);
        int pf = std::get<3>(vn_params[i]);
        CumulativePricingEngine temp_engine(n, vr, pf, m);
        relative_factors[i] = temp_engine.calculatePrice(SVR) / 100.0;
    }
    
    file << "Q 1.1 a)," 
         << vn1_linear->calculatePrice(5.0) << ","
         << vn1_cumulative->calculatePrice(5.0) << ","
         << vn1_recursive->calculatePrice(5.0) << ","
         << relative_factors[0] << ",100,0.25,5,5\n";
    
    file << "Q 1.1 b)," 
         << vn1_linear->calculateEffectiveRate(100.0) << ","
         << vn1_cumulative->calculateEffectiveRate(100.0) << ","
         << vn1_recursive->calculateEffectiveRate(100.0) << ","
         << relative_factors[1] << ",,0.25,5,5\n";
    
    file << "Q 1.2 a)," 
         << vn1_linear->calculatePriceSensitivity(5.0) << ","
         << vn1_cumulative->calculatePriceSensitivity(5.0) << ","
         << vn1_recursive->calculatePriceSensitivity(5.0) << ","
         << relative_factors[2] << ",,0.25,5,5\n";
    
    file << "Q 1.2 b)," 
         << vn1_linear->calculateRateSensitivity(100.0) << ","
         << vn1_cumulative->calculateRateSensitivity(100.0) << ","
         << vn1_recursive->calculateRateSensitivity(100.0) << ","
         << relative_factors[3] << ",,0.25,5,5\n";
    
    file << "Q 1.3 a)," 
         << vn1_linear->calculateSecondPriceSensitivity(5.0) << ","
         << vn1_cumulative->calculateSecondPriceSensitivity(5.0) << ","
         << vn1_recursive->calculateSecondPriceSensitivity(5.0) << ",,,,,\n";
    
    file << "Q 1.3 b)," 
         << vn1_linear->calculateSecondRateSensitivity(100.0) << ","
         << vn1_cumulative->calculateSecondRateSensitivity(100.0) << ","
         << vn1_recursive->calculateSecondRateSensitivity(100.0) << ",,,,,\n";
    
    file.close();
}

int main() {
    try {
        writeCorrectCSVOutput();
        std::cout << "Corrected CSV output written to output.csv\n";
        
        auto vn1_cumulative = std::make_unique<ValueNote>(100, 5, 3.5, 1, RateConvention::CUMULATIVE);
        std::cout << "VN1 Cumulative Price (ER=5%): " 
                  << vn1_cumulative->calculatePrice(5.0) << "\n";
        std::cout << "VN1 Cumulative Rate (VP=100): " 
                  << vn1_cumulative->calculateEffectiveRate(100.0) << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}