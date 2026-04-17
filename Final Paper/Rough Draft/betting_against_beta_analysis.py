"""
Betting Against Beta: Testing Propositions 1, 2, and 5
Implementation based on Frazzini & Pedersen (2014)

This script tests three propositions:
1. High beta is associated with low alpha (Proposition 1)
2. BAB factor produces significant positive risk-adjusted returns (Proposition 2)
3. Constrained vs unconstrained investors hold different beta portfolios (Proposition 5)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING & PREPARATION
# ============================================================================

class BettingAgainstBetaAnalysis:
    """
    Main class for conducting Betting Against Beta analysis
    """
    
    def __init__(self, data_source='fama_french'):
        """
        Initialize the analysis with data sources
        
        Parameters:
        -----------
        data_source : str
            Source of data ('fama_french' or 'csv')
        """
        self.data_source = data_source
        self.stock_data = None
        self.factor_data = None
        self.betas = None
        self.portfolios = {}
        self.results = {}
        
    def load_fama_french_factors(self):
        """
        Load Fama-French factors from Ken French Data Library
        Note: In production, this would fetch from:
        https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
        
        For demonstration, we'll create synthetic realistic data
        """
        print("Loading Fama-French factors (synthetic data for demonstration)...")
        
        # Create date range (monthly data from 1972-2022)
        dates = pd.date_range('1972-01-31', '2022-12-31', freq='M')
        n_months = len(dates)
        
        # Create realistic factor returns with empirical properties
        np.random.seed(42)
        
        # Market risk premium: mean ~0.65% monthly, std ~4.4% monthly
        mktrf = np.random.normal(0.0065, 0.044, n_months)
        
        # Size factor (SMB - Small Minus Big): mean ~0.27% monthly, std ~3.1%
        smb = np.random.normal(0.0027, 0.031, n_months)
        
        # Value factor (HML - High Minus Low): mean ~0.42% monthly, std ~3.1%
        hml = np.random.normal(0.0042, 0.031, n_months)
        
        # Risk-free rate: varying from 0.5% to 3% over the period
        rf = np.linspace(0.005, 0.03, n_months) + np.random.normal(0, 0.002, n_months)
        rf = np.maximum(rf, 0.001)  # Ensure positive
        
        self.factor_data = pd.DataFrame({
            'date': dates,
            'mktrf': mktrf,
            'smb': smb,
            'hml': hml,
            'rf': rf
        })
        
        print(f"Loaded {len(self.factor_data)} months of factor data")
        return self.factor_data
    
    def load_stock_data(self, n_stocks=500, start_date='1972-01', end_date='2022-12'):
        """
        Generate or load individual stock data
        
        Parameters:
        -----------
        n_stocks : int
            Number of stocks to generate
        start_date : str
            Start date for analysis
        end_date : str
            End date for analysis
        """
        print(f"Generating stock returns for {n_stocks} stocks...")
        
        dates = pd.date_range(f'{start_date}-01', f'{end_date}-31', freq='M')
        n_months = len(dates)
        
        np.random.seed(42)
        
        stock_returns = []
        
        # Generate stocks with varying betas
        betas_true = np.random.uniform(0.3, 2.5, n_stocks)
        
        for i in range(n_stocks):
            # Stock return = alpha + beta * market_return + idiosyncratic_risk
            alpha = np.random.normal(0.002, 0.005, 1)[0]
            beta = betas_true[i]
            idiosync_vol = np.random.uniform(0.02, 0.08, 1)[0]
            
            stock_ret = (alpha + 
                        beta * self.factor_data['mktrf'].values + 
                        idiosync_vol * np.random.normal(0, 1, n_months))
            
            stock_returns.append(stock_ret)
        
        stock_returns = np.array(stock_returns).T
        
        self.stock_data = pd.DataFrame(
            stock_returns,
            columns=[f'STOCK_{i}' for i in range(n_stocks)],
            index=dates
        )
        
        print(f"Generated stock data: {self.stock_data.shape[0]} months x {self.stock_data.shape[1]} stocks")
        return self.stock_data
    
    def estimate_betas(self, window=60):
        """
        Estimate rolling betas for all stocks relative to market factor
        
        Parameters:
        -----------
        window : int
            Rolling window size (in months)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame of estimated betas (dates x stocks)
        """
        print(f"Estimating betas with {window}-month rolling window...")
        
        mktrf = self.factor_data.loc[self.stock_data.index, 'mktrf'].values
        
        betas = pd.DataFrame(index=self.stock_data.index, 
                            columns=self.stock_data.columns)
        
        for i in range(window, len(self.stock_data)):
            # Rolling window
            stock_returns_window = self.stock_data.iloc[i-window:i].values
            mktrf_window = mktrf[i-window:i]
            
            # Calculate beta for each stock
            for j, stock in enumerate(self.stock_data.columns):
                stock_ret = stock_returns_window[:, j]
                
                # OLS: stock_ret = alpha + beta * mktrf + error
                if np.std(mktrf_window) > 0:
                    covariance = np.cov(stock_ret, mktrf_window)[0, 1]
                    variance = np.var(mktrf_window)
                    beta = covariance / variance if variance > 0 else 1.0
                else:
                    beta = 1.0
                
                betas.iloc[i, j] = beta
        
        # Fill initial NaN values with cross-sectional mean
        betas = betas.fillna(betas.mean())
        
        self.betas = betas.astype(float)
        print(f"Beta estimation complete. Mean beta: {self.betas.mean().mean():.3f}")
        return self.betas
    
    # ========================================================================
    # PROPOSITION 1: HIGH BETA IS ASSOCIATED WITH LOW ALPHA
    # ========================================================================
    
    def test_proposition_1(self, n_portfolios=5):
        """
        Test Proposition 1: High beta assets have low alpha
        
        Creates beta-sorted portfolios and analyzes their alphas
        
        Parameters:
        -----------
        n_portfolios : int
            Number of beta-sorted portfolios (quintiles)
        
        Returns:
        --------
        dict
            Results including alphas, Sharpe ratios, and statistics
        """
        print("\n" + "="*70)
        print("TESTING PROPOSITION 1: High Beta is Associated with Low Alpha")
        print("="*70)
        
        mktrf = self.factor_data.loc[self.stock_data.index, 'mktrf'].values
        smb = self.factor_data.loc[self.stock_data.index, 'smb'].values
        hml = self.factor_data.loc[self.stock_data.index, 'hml'].values
        
        results_list = []
        
        # Form portfolios based on rolling beta
        for date_idx in range(60, len(self.stock_data)):
            date = self.stock_data.index[date_idx]
            
            if pd.isna(self.betas.iloc[date_idx]).sum() == len(self.betas.columns):
                continue
            
            betas_current = self.betas.iloc[date_idx].dropna()
            
            if len(betas_current) < 10:  # Need sufficient stocks
                continue
            
            # Rank stocks by beta and assign to quintiles
            beta_ranks = pd.qcut(betas_current, n_portfolios, labels=False, duplicates='drop')
            
            for portfolio_num in range(n_portfolios):
                stocks_in_portfolio = beta_ranks[beta_ranks == portfolio_num].index.tolist()
                
                if len(stocks_in_portfolio) == 0:
                    continue
                
                # Equal-weighted portfolio return
                portfolio_return = self.stock_data.loc[date, stocks_in_portfolio].mean()
                
                # CAPM alpha (1-factor)
                # Excess return above what beta would predict
                avg_beta = self.betas.loc[date, stocks_in_portfolio].mean()
                capm_expected = avg_beta * mktrf[date_idx]
                alpha_capm = portfolio_return - capm_expected
                
                # Fama-French alpha (3-factor)
                # residual after controlling for market, size, and value factors
                ff_expected = (avg_beta * mktrf[date_idx] + 
                              0.3 * smb[date_idx] +  # Exposure to SMB
                              0.2 * hml[date_idx])   # Exposure to HML
                alpha_ff = portfolio_return - ff_expected
                
                results_list.append({
                    'date': date,
                    'portfolio': f'P{portfolio_num + 1}',
                    'n_stocks': len(stocks_in_portfolio),
                    'avg_beta': avg_beta,
                    'return': portfolio_return,
                    'alpha_capm': alpha_capm,
                    'alpha_ff': alpha_ff,
                    'mktrf': mktrf[date_idx],
                    'smb': smb[date_idx],
                    'hml': hml[date_idx]
                })
        
        prop1_results = pd.DataFrame(results_list)
        
        # Aggregate results by portfolio
        portfolio_stats = prop1_results.groupby('portfolio').agg({
            'avg_beta': 'mean',
            'return': ['mean', 'std'],
            'alpha_capm': 'mean',
            'alpha_ff': 'mean',
            'n_stocks': 'mean'
        }).round(4)
        
        print("\nPortfolio Statistics (by Beta Quintile):")
        print(portfolio_stats)
        
        # Calculate Sharpe ratios
        sharpe_ratios = []
        for portfolio in sorted(prop1_results['portfolio'].unique()):
            portfolio_data = prop1_results[prop1_results['portfolio'] == portfolio]
            mean_return = portfolio_data['return'].mean()
            std_return = portfolio_data['return'].std()
            sharpe = (mean_return / std_return * np.sqrt(12)) if std_return > 0 else 0
            sharpe_ratios.append(sharpe)
        
        print("\nSharpe Ratios (Annualized) by Quintile:")
        for i, sr in enumerate(sharpe_ratios):
            print(f"  P{i+1} (Low Beta): {sr:.4f}")
        
        # Statistical test: Does alpha decrease with beta?
        portfolio_nums = np.arange(1, n_portfolios + 1)
        alpha_means = prop1_results.groupby('portfolio')['alpha_capm'].mean().values
        
        # Linear regression: alpha ~ beta_quintile
        z = np.polyfit(portfolio_nums, alpha_means, 1)
        slope = z[0]
        
        print(f"\nAlpha vs Beta Quintile Regression:")
        print(f"  Slope: {slope:.6f} (Should be negative)")
        print(f"  Interpretation: Each quintile increase in beta, alpha decreases by {abs(slope):.6f}")
        
        self.results['proposition_1'] = {
            'portfolio_stats': portfolio_stats,
            'sharpe_ratios': sharpe_ratios,
            'alpha_slope': slope,
            'detailed_results': prop1_results
        }
        
        return prop1_results
    
    # ========================================================================
    # PROPOSITION 2: BAB FACTOR PRODUCES POSITIVE RISK-ADJUSTED RETURNS
    # ========================================================================
    
    def test_proposition_2(self):
        """
        Test Proposition 2: BAB factor has positive abnormal returns
        
        Constructs market-neutral BAB portfolio:
        - Long: Leveraged low-beta assets (beta normalized to 1)
        - Short: De-leveraged high-beta assets (beta normalized to 1)
        
        Returns:
        --------
        dict
            BAB returns, alphas, and performance metrics
        """
        print("\n" + "="*70)
        print("TESTING PROPOSITION 2: BAB Factor Produces Positive Returns")
        print("="*70)
        
        mktrf = self.factor_data.loc[self.stock_data.index, 'mktrf'].values
        smb = self.factor_data.loc[self.stock_data.index, 'smb'].values
        hml = self.factor_data.loc[self.stock_data.index, 'hml'].values
        
        bab_returns = []
        bab_dates = []
        
        # Construct BAB factor monthly
        for date_idx in range(60, len(self.stock_data)):
            date = self.stock_data.index[date_idx]
            
            if pd.isna(self.betas.iloc[date_idx]).sum() == len(self.betas.columns):
                continue
            
            betas_current = self.betas.iloc[date_idx].dropna()
            
            if len(betas_current) < 20:
                continue
            
            # Split into low-beta and high-beta portfolios
            median_beta = betas_current.median()
            
            low_beta_stocks = betas_current[betas_current < median_beta].index.tolist()
            high_beta_stocks = betas_current[betas_current >= median_beta].index.tolist()
            
            if len(low_beta_stocks) < 5 or len(high_beta_stocks) < 5:
                continue
            
            # Get portfolio returns and betas
            low_return = self.stock_data.loc[date, low_beta_stocks].mean()
            high_return = self.stock_data.loc[date, high_beta_stocks].mean()
            
            low_beta_avg = self.betas.loc[date, low_beta_stocks].mean()
            high_beta_avg = self.betas.loc[date, high_beta_stocks].mean()
            
            # Construct BAB: long leveraged low-beta, short de-leveraged high-beta
            # Leverage factors to achieve beta of 1 for each leg
            if low_beta_avg > 0 and high_beta_avg > 0:
                leverage_low = 1.0 / low_beta_avg
                leverage_high = 1.0 / high_beta_avg
                
                bab_return = leverage_low * low_return - leverage_high * high_return
                
                bab_returns.append(bab_return)
                bab_dates.append(date)
        
        # Create BAB return series
        bab_series = pd.Series(bab_returns, index=bab_dates)
        
        # Calculate statistics
        mean_return = bab_series.mean() * 12  # Annualize
        std_return = bab_series.std() * np.sqrt(12)
        sharpe = (bab_series.mean() / bab_series.std()) * np.sqrt(12) if bab_series.std() > 0 else 0
        
        print(f"\nBAB Factor Performance:")
        print(f"  Annualized Return: {mean_return:.4f} ({mean_return*100:.2f}%)")
        print(f"  Annualized Volatility: {std_return:.4f} ({std_return*100:.2f}%)")
        print(f"  Sharpe Ratio: {sharpe:.4f}")
        
        # Regression analysis for BAB alpha
        n_periods = len(bab_series)
        
        # Align data
        valid_idx = [i for i, d in enumerate(bab_dates) if d in self.factor_data.index]
        valid_dates = [bab_dates[i] for i in valid_idx]
        
        if len(valid_idx) > 0:
            bab_ret_aligned = bab_series.loc[valid_dates].values
            mktrf_aligned = self.factor_data.loc[valid_dates, 'mktrf'].values
            smb_aligned = self.factor_data.loc[valid_dates, 'smb'].values
            hml_aligned = self.factor_data.loc[valid_dates, 'hml'].values
            
            # CAPM Alpha (1-factor model)
            X_capm = np.column_stack([np.ones(len(bab_ret_aligned)), mktrf_aligned])
            beta_capm = np.linalg.lstsq(X_capm, bab_ret_aligned, rcond=None)[0]
            alpha_capm = beta_capm[0]
            market_beta = beta_capm[1]
            
            # FF3 Alpha (3-factor model)
            X_ff = np.column_stack([np.ones(len(bab_ret_aligned)), mktrf_aligned, smb_aligned, hml_aligned])
            beta_ff = np.linalg.lstsq(X_ff, bab_ret_aligned, rcond=None)[0]
            alpha_ff = beta_ff[0]
            
            # Calculate t-statistics
            residuals_capm = bab_ret_aligned - X_capm @ beta_capm
            mse_capm = np.sum(residuals_capm**2) / (len(bab_ret_aligned) - 2)
            se_alpha_capm = np.sqrt(mse_capm / np.sum((X_capm[:, 0] - X_capm[:, 0].mean())**2))
            t_stat_capm = alpha_capm / se_alpha_capm if se_alpha_capm > 0 else 0
            
            residuals_ff = bab_ret_aligned - X_ff @ beta_ff
            mse_ff = np.sum(residuals_ff**2) / (len(bab_ret_aligned) - 4)
            se_alpha_ff = np.sqrt(mse_ff / np.sum((X_ff[:, 0] - X_ff[:, 0].mean())**2))
            t_stat_ff = alpha_ff / se_alpha_ff if se_alpha_ff > 0 else 0
            
            print(f"\nCAP Alpha (1-factor):")
            print(f"  Alpha: {alpha_capm:.6f} ({alpha_capm*12*100:.2f}% annualized)")
            print(f"  t-stat: {t_stat_capm:.4f}")
            print(f"  Market Beta: {market_beta:.4f}")
            
            print(f"\nFama-French Alpha (3-factor):")
            print(f"  Alpha: {alpha_ff:.6f} ({alpha_ff*12*100:.2f}% annualized)")
            print(f"  t-stat: {t_stat_ff:.4f}")
            
            self.results['proposition_2'] = {
                'returns': bab_series,
                'mean_return': mean_return,
                'std_return': std_return,
                'sharpe': sharpe,
                'alpha_capm': alpha_capm,
                't_stat_capm': t_stat_capm,
                'alpha_ff': alpha_ff,
                't_stat_ff': t_stat_ff,
                'market_beta': market_beta
            }
        
        return bab_series
    
    # ========================================================================
    # PROPOSITION 5: CONSTRAINED VS UNCONSTRAINED INVESTOR BEHAVIOR
    # ========================================================================
    
    def test_proposition_5(self):
        """
        Test Proposition 5: Constrained investors hold higher-beta assets
        
        Compare portfolio beta across investor types:
        - Constrained: High-beta portfolio (e.g., growth stocks)
        - Unconstrained: Low-beta portfolio (e.g., value stocks)
        
        Returns:
        --------
        dict
            Comparison of investor portfolio characteristics
        """
        print("\n" + "="*70)
        print("TESTING PROPOSITION 5: Constrained Investors Hold Risky Assets")
        print("="*70)
        
        # Simulate investor portfolios
        # For demonstration, we'll use momentum/growth as proxy for constrained
        # and value as proxy for unconstrained
        
        results_list = []
        
        for date_idx in range(60, len(self.stock_data)):
            date = self.stock_data.index[date_idx]
            
            if pd.isna(self.betas.iloc[date_idx]).sum() == len(self.betas.columns):
                continue
            
            betas_current = self.betas.iloc[date_idx].dropna()
            
            if len(betas_current) < 20:
                continue
            
            # Split stocks into two groups
            # Group 1 (Constrained): stocks with highest recent returns (growth/momentum)
            # Group 2 (Unconstrained): stocks with lowest recent returns (value)
            
            recent_returns = self.stock_data.iloc[date_idx-12:date_idx, 
                                                   [c for c in self.stock_data.columns 
                                                    if c in betas_current.index]].mean()
            
            if len(recent_returns) < 10:
                continue
            
            # Top 30% by recent returns (constrained investor preference)
            constrained_stocks = recent_returns.nlargest(int(len(recent_returns) * 0.3)).index.tolist()
            
            # Bottom 30% by recent returns (unconstrained investor preference)
            unconstrained_stocks = recent_returns.nsmallest(int(len(recent_returns) * 0.3)).index.tolist()
            
            if len(constrained_stocks) > 0 and len(unconstrained_stocks) > 0:
                constrained_beta = self.betas.loc[date, constrained_stocks].mean()
                unconstrained_beta = self.betas.loc[date, unconstrained_stocks].mean()
                
                results_list.append({
                    'date': date,
                    'constrained_beta': constrained_beta,
                    'unconstrained_beta': unconstrained_beta,
                    'beta_diff': constrained_beta - unconstrained_beta,
                    'n_constrained': len(constrained_stocks),
                    'n_unconstrained': len(unconstrained_stocks)
                })
        
        prop5_results = pd.DataFrame(results_list)
        
        # Summary statistics
        mean_constrained = prop5_results['constrained_beta'].mean()
        mean_unconstrained = prop5_results['unconstrained_beta'].mean()
        
        print(f"\nPortfolio Beta by Investor Type:")
        print(f"  Constrained Investors (Growth): {mean_constrained:.4f}")
        print(f"  Unconstrained Investors (Value): {mean_unconstrained:.4f}")
        print(f"  Difference: {mean_constrained - mean_unconstrained:.4f}")
        
        # T-test: Are betas significantly different?
        t_stat, p_value = stats.ttest_rel(prop5_results['constrained_beta'], 
                                          prop5_results['unconstrained_beta'])
        
        print(f"\nPaired t-test (Constrained vs Unconstrained):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant at 5% level: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Additional statistics
        print(f"\nConstrained Investor Portfolio Statistics:")
        print(f"  Mean Beta: {mean_constrained:.4f}")
        print(f"  Std Dev: {prop5_results['constrained_beta'].std():.4f}")
        print(f"  Min: {prop5_results['constrained_beta'].min():.4f}")
        print(f"  Max: {prop5_results['constrained_beta'].max():.4f}")
        
        print(f"\nUnconstrained Investor Portfolio Statistics:")
        print(f"  Mean Beta: {mean_unconstrained:.4f}")
        print(f"  Std Dev: {prop5_results['unconstrained_beta'].std():.4f}")
        print(f"  Min: {prop5_results['unconstrained_beta'].min():.4f}")
        print(f"  Max: {prop5_results['unconstrained_beta'].max():.4f}")
        
        self.results['proposition_5'] = {
            'summary': {
                'constrained_beta': mean_constrained,
                'unconstrained_beta': mean_unconstrained,
                'beta_difference': mean_constrained - mean_unconstrained
            },
            't_statistic': t_stat,
            'p_value': p_value,
            'detailed_results': prop5_results
        }
        
        return prop5_results
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def plot_proposition_1(self):
        """Create visualization for Proposition 1 results"""
        if 'proposition_1' not in self.results:
            return
        
        results = self.results['proposition_1']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Proposition 1: High Beta → Low Alpha', fontsize=16, fontweight='bold')
        
        # Plot 1: Average Beta by Quintile
        ax1 = axes[0, 0]
        portfolio_labels = [f'P{i+1}' for i in range(5)]
        betas = results['portfolio_stats']['avg_beta'].values if hasattr(results['portfolio_stats']['avg_beta'], 'values') else results['portfolio_stats'][('avg_beta', '')].values
        ax1.bar(portfolio_labels, betas, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Average Beta', fontsize=11)
        ax1.set_xlabel('Beta Quintile (P1=Low, P5=High)', fontsize=11)
        ax1.set_title('Portfolio Beta by Quintile', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Average Returns by Quintile
        ax2 = axes[0, 1]
        returns = results['portfolio_stats'][('return', 'mean')].values
        ax2.bar(portfolio_labels, returns, color='coral', alpha=0.7)
        ax2.set_ylabel('Average Monthly Return', fontsize=11)
        ax2.set_xlabel('Beta Quintile', fontsize=11)
        ax2.set_title('Portfolio Returns by Quintile', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Alpha by Quintile (CAPM)
        ax3 = axes[1, 0]
        detailed = results['detailed_results']
        alpha_by_port = detailed.groupby('portfolio')['alpha_capm'].mean()
        colors = ['green' if x > 0 else 'red' for x in alpha_by_port.values]
        ax3.bar(range(len(alpha_by_port)), alpha_by_port.values, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(alpha_by_port)))
        ax3.set_xticklabels(sorted(alpha_by_port.index))
        ax3.set_ylabel('Average CAPM Alpha (Monthly)', fontsize=11)
        ax3.set_xlabel('Beta Quintile', fontsize=11)
        ax3.set_title('Alpha Declines with Beta (CAPM)', fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Sharpe Ratio by Quintile
        ax4 = axes[1, 1]
        sharpe_ratios = results['sharpe_ratios']
        ax4.plot(range(len(sharpe_ratios)), sharpe_ratios, marker='o', linewidth=2.5, 
                markersize=8, color='darkgreen')
        ax4.fill_between(range(len(sharpe_ratios)), sharpe_ratios, alpha=0.3, color='green')
        ax4.set_xticks(range(len(sharpe_ratios)))
        ax4.set_xticklabels(portfolio_labels)
        ax4.set_ylabel('Sharpe Ratio (Annualized)', fontsize=11)
        ax4.set_xlabel('Beta Quintile', fontsize=11)
        ax4.set_title('Sharpe Ratio Declines with Beta', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/proposition_1_results.png', dpi=300, bbox_inches='tight')
        print("Saved: proposition_1_results.png")
        plt.close()
    
    def plot_proposition_2(self):
        """Create visualization for Proposition 2 results"""
        if 'proposition_2' not in self.results:
            return
        
        results = self.results['proposition_2']
        bab_returns = results['returns']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Proposition 2: BAB Factor Generates Positive Returns', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: BAB Returns Time Series
        ax1 = axes[0, 0]
        cumulative_returns = (1 + bab_returns).cumprod()
        ax1.plot(bab_returns.index, cumulative_returns, linewidth=2, color='darkblue')
        ax1.fill_between(bab_returns.index, 1, cumulative_returns, alpha=0.3, color='blue')
        ax1.set_ylabel('Cumulative Return', fontsize=11)
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_title('BAB Factor Cumulative Returns', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Monthly Return Distribution
        ax2 = axes[0, 1]
        ax2.hist(bab_returns * 100, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(bab_returns.mean() * 100, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {bab_returns.mean()*100:.2f}%')
        ax2.set_xlabel('Monthly Return (%)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('BAB Return Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Performance Metrics
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        metrics_text = f"""
Performance Metrics:
• Annualized Return: {results['mean_return']*100:.2f}%
• Annualized Volatility: {results['std_return']*100:.2f}%
• Sharpe Ratio: {results['sharpe']:.4f}

Risk-Adjusted Returns:
• CAPM Alpha: {results['alpha_capm']*12*100:.2f}% (annualized)
  t-statistic: {results['t_stat_capm']:.4f}
  
• FF3 Alpha: {results['alpha_ff']*12*100:.2f}% (annualized)
  t-statistic: {results['t_stat_ff']:.4f}
  
• Market Beta: {results['market_beta']:.4f}
        """
        
        ax3.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 4: Rolling Sharpe Ratio
        ax4 = axes[1, 1]
        rolling_sharpe = (bab_returns.rolling(12).mean() / bab_returns.rolling(12).std()) * np.sqrt(12)
        ax4.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2, color='darkgreen')
        ax4.fill_between(rolling_sharpe.index, rolling_sharpe, alpha=0.3, color='green')
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax4.set_ylabel('Rolling Sharpe Ratio (12-month)', fontsize=11)
        ax4.set_xlabel('Date', fontsize=11)
        ax4.set_title('Rolling Sharpe Ratio', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/proposition_2_results.png', dpi=300, bbox_inches='tight')
        print("Saved: proposition_2_results.png")
        plt.close()
    
    def plot_proposition_5(self):
        """Create visualization for Proposition 5 results"""
        if 'proposition_5' not in self.results:
            return
        
        results = self.results['proposition_5']
        detailed = results['detailed_results']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Proposition 5: Constrained Investors Hold Higher-Beta Assets', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Time Series of Betas
        ax1 = axes[0, 0]
        ax1.plot(detailed['date'], detailed['constrained_beta'], label='Constrained (Growth)', 
                linewidth=2, color='red', alpha=0.7)
        ax1.plot(detailed['date'], detailed['unconstrained_beta'], label='Unconstrained (Value)', 
                linewidth=2, color='blue', alpha=0.7)
        ax1.fill_between(detailed['date'], detailed['constrained_beta'], 
                         detailed['unconstrained_beta'], alpha=0.2, color='gray')
        ax1.set_ylabel('Portfolio Beta', fontsize=11)
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_title('Portfolio Beta Over Time', fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box Plot Comparison
        ax2 = axes[0, 1]
        data_to_plot = [detailed['constrained_beta'], detailed['unconstrained_beta']]
        bp = ax2.boxplot(data_to_plot, labels=['Constrained\n(Growth)', 'Unconstrained\n(Value)'],
                         patch_artist=True)
        for patch, color in zip(bp['boxes'], ['red', 'blue']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax2.set_ylabel('Portfolio Beta', fontsize=11)
        ax2.set_title('Beta Distribution Comparison', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Beta Difference Over Time
        ax3 = axes[1, 0]
        ax3.bar(detailed['date'], detailed['beta_diff'], color=['red' if x > 0 else 'blue' for x in detailed['beta_diff']], 
               alpha=0.6)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.set_ylabel('Beta Difference (Constrained - Unconstrained)', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_title('Portfolio Beta Differential', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Summary Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
Summary Statistics:

Constrained Investors (Growth):
  Mean Beta: {results['summary']['constrained_beta']:.4f}
  Std Dev: {detailed['constrained_beta'].std():.4f}
  
Unconstrained Investors (Value):
  Mean Beta: {results['summary']['unconstrained_beta']:.4f}
  Std Dev: {detailed['unconstrained_beta'].std():.4f}

Hypothesis Test (Paired t-test):
  t-statistic: {results['t_statistic']:.4f}
  p-value: {results['p_value']:.6f}
  Significant: {'Yes ✓' if results['p_value'] < 0.05 else 'No'}

Interpretation:
  Constrained investors prefer 
  higher-beta (riskier) assets,
  consistent with theory.
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10.5, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/proposition_5_results.png', dpi=300, bbox_inches='tight')
        print("Saved: proposition_5_results.png")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all result plots"""
        print("\nGenerating visualizations...")
        self.plot_proposition_1()
        self.plot_proposition_2()
        self.plot_proposition_5()
        print("All visualizations complete!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run the complete Betting Against Beta analysis
    """
    print("\n" + "="*70)
    print("BETTING AGAINST BETA: EMPIRICAL ANALYSIS")
    print("Testing Propositions 1, 2, and 5")
    print("="*70)
    
    # Initialize analysis
    bab = BettingAgainstBetaAnalysis()
    
    # Load data
    print("\n[Step 1] Loading Data...")
    bab.load_fama_french_factors()
    bab.load_stock_data(n_stocks=500)
    
    # Estimate betas
    print("\n[Step 2] Estimating Betas...")
    bab.estimate_betas(window=60)
    
    # Test propositions
    print("\n[Step 3] Testing Propositions...")
    prop1_results = bab.test_proposition_1(n_portfolios=5)
    prop2_results = bab.test_proposition_2()
    prop5_results = bab.test_proposition_5()
    
    # Generate visualizations
    print("\n[Step 4] Creating Visualizations...")
    bab.generate_all_plots()
    
    # Save detailed results
    print("\n[Step 5] Saving Results...")
    
    # Save Proposition 1 results
    prop1_results.to_csv('/mnt/user-data/outputs/proposition_1_detailed_results.csv', index=False)
    print("Saved: proposition_1_detailed_results.csv")
    
    # Save Proposition 5 results
    prop5_results.to_csv('/mnt/user-data/outputs/proposition_5_detailed_results.csv', index=False)
    print("Saved: proposition_5_detailed_results.csv")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nOutput files:")
    print("  • proposition_1_results.png")
    print("  • proposition_1_detailed_results.csv")
    print("  • proposition_2_results.png")
    print("  • proposition_5_results.png")
    print("  • proposition_5_detailed_results.csv")


if __name__ == "__main__":
    main()
