import pandas as pd
import numpy as np
from collections import defaultdict, deque
import random
import warnings
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

def backtester_without_TC(weights_df):
    data = pd.read_csv('cross_val_data.csv')
    data = data.iloc[:, 1:]  

    weights_df = weights_df.fillna(0)

    # Use appropriate data range
    total_rows = len(weights_df)
    start_date = max(0, total_rows - 500)
    end_date = total_rows - 1
    
    # Ensure bounds
    start_date = min(start_date, total_rows - 1)
    end_date = min(end_date, total_rows - 1)
    
    if start_date >= end_date:
        start_date = max(0, total_rows - 100)
        end_date = total_rows - 1

    initial_notional = 1
    df_returns = pd.DataFrame()

    for i in range(0, 20):
        data_symbol = data[data['Symbol'] == i]
        data_symbol = data_symbol['Close']
        data_symbol = data_symbol.reset_index(drop=True)   
        data_symbol = data_symbol / data_symbol.shift(1) - 1
        df_returns = pd.concat([df_returns, data_symbol], axis=1, ignore_index=True)
    
    df_returns = df_returns.fillna(0)
    
    # Align dataframes
    weights_subset = weights_df.loc[start_date:end_date]    
    returns_subset = df_returns.loc[start_date:end_date]
    
    common_indices = weights_subset.index.intersection(returns_subset.index)
    weights_subset = weights_subset.loc[common_indices]
    returns_subset = returns_subset.loc[common_indices]
    
    portfolio_returns = weights_subset.mul(returns_subset)

    notional = initial_notional
    returns = []

    for date in common_indices:
        daily_return = portfolio_returns.loc[date].sum()
        returns.append(daily_return)
        notional = notional * (1 + daily_return)

    if len(returns) == 0:
        return [0.0, 0.0]
    
    net_return = ((notional - initial_notional) / initial_notional) * 100
    returns_series = pd.Series(returns)
    sharpe_ratio = returns_series.mean() / returns_series.std() if returns_series.std() != 0 else 0
    return [net_return, sharpe_ratio]


def backtester_with_TC(weights_df):
    """Fixed backtester with proper transaction cost handling"""
    data = pd.read_csv('train_data.csv')
    data = data.iloc[:, 1:]

    weights_df = weights_df.fillna(0)

    total_rows = len(weights_df)
    start_date = max(0, total_rows - 500)
    end_date = total_rows - 1
    
    start_date = min(start_date, total_rows - 1)
    end_date = min(end_date, total_rows - 1)
    
    if start_date >= end_date:
        start_date = max(0, total_rows - 100)
        end_date = total_rows - 1

    initial_notional = 1.0
    df_returns = pd.DataFrame()

    for i in range(0, 20):
        data_symbol = data[data['Symbol'] == i]
        data_symbol = data_symbol['Close']
        data_symbol = data_symbol.reset_index(drop=True)   
        data_symbol = data_symbol / data_symbol.shift(1) - 1
        df_returns = pd.concat([df_returns, data_symbol], axis=1, ignore_index=True)
    
    df_returns = df_returns.fillna(0)
    
    weights_subset = weights_df.loc[start_date:end_date]    
    returns_subset = df_returns.loc[start_date:end_date]
    
    common_indices = weights_subset.index.intersection(returns_subset.index)
    weights_subset = weights_subset.loc[common_indices]
    returns_subset = returns_subset.loc[common_indices]
    
    portfolio_returns = weights_subset.mul(returns_subset)

    notional = initial_notional
    daily_returns = []  
    prev_weights = None
    total_transaction_costs = 0

    for idx, date in enumerate(common_indices):
        # Calculate gross portfolio return (before transaction costs)
        gross_daily_return = portfolio_returns.loc[date].sum()
        
        # Calculate transaction costs for this day
        current_weights = weights_subset.loc[date]
        daily_tc = 0
        
        if prev_weights is not None:
            daily_turnover = abs(current_weights - prev_weights).sum()
            daily_tc = daily_turnover * 0.01  # 1% transaction cost
            total_transaction_costs += daily_tc
        
        # TC reduces the notional before applying returns
        tc_drag = daily_tc / notional if notional > 0 else 0
        net_daily_return = gross_daily_return - tc_drag
        
        daily_returns.append(net_daily_return)
        
        # Update notional
        notional = notional * (1 + net_daily_return)
        prev_weights = current_weights.copy()

    if len(daily_returns) == 0:
        return [0.0, 0.0]

    returns_series = pd.Series(daily_returns)
    
    # Net return calculation
    net_return = ((notional - initial_notional) / initial_notional) * 100
    
    # Sharpe ratio calculation using the SAME returns that created net_return
    mean_return = returns_series.mean()
    std_return = returns_series.std()
    sharpe_ratio = mean_return / std_return if std_return != 0 else 0
    
    print(f"Debug Transaction Cost Backtester:")
    print(f"  Total days: {len(daily_returns)}")
    print(f"  Mean daily return: {mean_return:.6f}")
    print(f"  Std daily return: {std_return:.6f}")
    print(f"  Total transaction costs: {total_transaction_costs:.4f}")
    print(f"  Final notional: {notional:.4f}")
    print(f"  Net return: {net_return:.4f}%")
    print(f"  Sharpe ratio: {sharpe_ratio:.6f}")
    print(f"  Negative return days: {(returns_series < 0).sum()}/{len(returns_series)}")
    
    return [net_return, sharpe_ratio]

class TradingStrategies:
    def __init__(self, data):
        # Remove first column if it exists
        if isinstance(data, pd.DataFrame) and len(data.columns) > 0:
            first_col = data.columns[0]
            if str(first_col).startswith('Unnamed') or first_col == 0:
                self.data = data.iloc[:, 1:]
            else:
                self.data = data.copy()
        else:
            self.data = data.copy()
            
        self.data = self.data.reset_index(drop=True)
        
        required_cols = ['Symbol', 'Close', 'High', 'Low']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            print(f"Available columns: {self.data.columns.tolist()}")
        
        self.symbols = sorted(self.data['Symbol'].unique()) if 'Symbol' in self.data.columns else list(range(20))
        
        if 'Date' in self.data.columns:
            self.data = self.data.sort_values(['Date', 'Symbol']).reset_index(drop=True)
        
    def prepare_data(self):
        """Prepare data in pivot format for easier calculations"""
        try:
            if 'Date' in self.data.columns:
                unique_dates = sorted(self.data['Date'].unique())
                date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
                self.data['row_idx'] = self.data['Date'].map(date_to_idx)
            else:
                n_symbols = len(self.symbols)
                self.data['row_idx'] = self.data.index // n_symbols
            
            # Create pivot tables
            self.close_prices = self.data.pivot(index='row_idx', columns='Symbol', values='Close')
            self.high_prices = self.data.pivot(index='row_idx', columns='Symbol', values='High')
            self.low_prices = self.data.pivot(index='row_idx', columns='Symbol', values='Low')
            
            if 'Volume' in self.data.columns:
                self.volume_data = self.data.pivot(index='row_idx', columns='Symbol', values='Volume')
            else:
                self.volume_data = pd.DataFrame()
            
            self.close_prices = self.close_prices.fillna(method='ffill').fillna(method='bfill')
            self.high_prices = self.high_prices.fillna(method='ffill').fillna(method='bfill')
            self.low_prices = self.low_prices.fillna(method='ffill').fillna(method='bfill')
            
            if not self.volume_data.empty:
                self.volume_data = self.volume_data.fillna(method='ffill').fillna(method='bfill')
                
            self.dates = list(range(len(self.close_prices)))
                
            print(f"Data prepared successfully. Close prices shape: {self.close_prices.shape}")
            print(f"Available symbols: {sorted(self.close_prices.columns.tolist())}")
            
        except Exception as e:
            print(f"Error in prepare_data: {e}")
            print(f"Data columns: {self.data.columns.tolist()}")
            print(f"Data shape: {self.data.shape}")
            print(f"Sample data:")
            print(self.data.head())
            raise
    
    def task1_Strategy1(self, current_date_idx):
        """Strategy 1: Average Weekly Returns"""
        if current_date_idx < 250:
            return pd.Series(0, index=range(20))
        
        end_idx = current_date_idx
        start_idx = max(0, end_idx - 250)
        
        weekly_returns = {}
        
        for symbol in range(20):
            if symbol not in self.close_prices.columns:
                weekly_returns[symbol] = 0
                continue
                
            prices = self.close_prices.iloc[start_idx:end_idx][symbol].dropna()
            if len(prices) < 50:
                weekly_returns[symbol] = 0
                continue
                
            weekly_rets = []
            for week_start in range(5, len(prices), 5):
                if week_start < len(prices):
                    week_end_price = prices.iloc[week_start]
                    prev_week_end_price = prices.iloc[week_start - 5]
                    if prev_week_end_price != 0:
                        weekly_ret = (week_end_price - prev_week_end_price) / prev_week_end_price
                        weekly_rets.append(weekly_ret)
            
            weekly_returns[symbol] = np.mean(weekly_rets) if weekly_rets else 0
        
        returns_series = pd.Series(weekly_returns)
        ranked_stocks = returns_series.sort_values(ascending=False)
        
        weights = pd.Series(0.0, index=range(20))
        
        # Top 6 get negative weights
        top_6 = ranked_stocks.head(6).index
        weights[top_6] = -1/6
        
        # Bottom 6 get positive weights
        bottom_6 = ranked_stocks.tail(6).index
        weights[bottom_6] = 1/6
        
        return weights
    
    def task1_Strategy2(self, current_date_idx):
        """Strategy 2: Mean Reversion (SMA vs LMA)"""
        if current_date_idx < 30:
            return pd.Series(0, index=range(20))
        
        relative_positions = {}
        
        for symbol in range(20):
            if symbol not in self.close_prices.columns:
                relative_positions[symbol] = 0
                continue
                
            prices = self.close_prices.iloc[max(0, current_date_idx-30):current_date_idx][symbol].dropna()
            
            if len(prices) < 30:
                relative_positions[symbol] = 0
                continue
                
            lma = prices.tail(30).mean()
            sma = prices.tail(5).mean()
            
            relative_pos = (sma - lma) / lma if lma != 0 else 0
            relative_positions[symbol] = relative_pos
        
        rel_pos_series = pd.Series(relative_positions)
        ranked_stocks = rel_pos_series.sort_values(ascending=False)
        
        weights = pd.Series(0.0, index=range(20))
        
        # Top 5 get negative weights
        top_5 = ranked_stocks.head(5).index
        weights[top_5] = -1/5
        
        # Bottom 5 get positive weights
        bottom_5 = ranked_stocks.tail(5).index
        weights[bottom_5] = 1/5
        
        return weights
    
    def task1_Strategy3(self, current_date_idx):
        """Strategy 3: Rate of Change (ROC)"""
        if current_date_idx < 7:
            return pd.Series(0, index=range(20))
        
        roc_values = {}
        
        for symbol in range(20):
            if symbol not in self.close_prices.columns:
                roc_values[symbol] = 0
                continue
                
            current_price = self.close_prices.iloc[current_date_idx-1][symbol]
            price_7_days_ago = self.close_prices.iloc[current_date_idx-8][symbol]
            
            if pd.notna(current_price) and pd.notna(price_7_days_ago) and price_7_days_ago != 0:
                roc = 100 * (current_price - price_7_days_ago) / price_7_days_ago
                roc_values[symbol] = roc
            else:
                roc_values[symbol] = 0
        
        roc_series = pd.Series(roc_values)
        ranked_stocks = roc_series.sort_values(ascending=False)
        
        weights = pd.Series(0.0, index=range(20))
        
        # Top 5 get negative weights, bottom 5 get positive weights
        top_5 = ranked_stocks.head(5).index
        weights[top_5] = -1/5
        
        bottom_5 = ranked_stocks.tail(5).index
        weights[bottom_5] = 1/5
        
        return weights
    
    def task1_Strategy4(self, current_date_idx):
        """Strategy 4: Support and Resistance"""
        if current_date_idx < 21:
            return pd.Series(0, index=range(20))
        
        proximities = {}
        
        for symbol in range(20):
            if symbol not in self.close_prices.columns:
                proximities[symbol] = {'support': 0, 'resistance': 0}
                continue
                
            prices = self.close_prices.iloc[max(0, current_date_idx-21):current_date_idx][symbol].dropna()
            
            if len(prices) < 21:
                proximities[symbol] = {'support': 0, 'resistance': 0}
                continue
            
            sma_21 = prices.mean()
            std_21 = prices.std()
            
            resistance = sma_21 + 3 * std_21
            support = sma_21 - 3 * std_21
            current_price = prices.iloc[-1]
            
            prox_resistance = abs(current_price - resistance) / resistance if resistance != 0 else 0
            prox_support = abs(current_price - support) / support if support != 0 else 0
            
            proximities[symbol] = {'support': prox_support, 'resistance': prox_resistance}
        
        support_prox = {symbol: proximities[symbol]['support'] for symbol in range(20)}
        resistance_prox = {symbol: proximities[symbol]['resistance'] for symbol in range(20)}
        
        support_ranked = sorted(support_prox.items(), key=lambda x: x[1])
        top_4_support = [x[0] for x in support_ranked[:4]]
        
        remaining_stocks = [s for s in range(20) if s not in top_4_support]
        resistance_remaining = {s: resistance_prox[s] for s in remaining_stocks}
        resistance_ranked = sorted(resistance_remaining.items(), key=lambda x: x[1])
        top_4_resistance = [x[0] for x in resistance_ranked[:4]]
        
        weights = pd.Series(0.0, index=range(20))
        
        for symbol in top_4_support:
            weights[symbol] = 1/4
            
        for symbol in top_4_resistance:
            weights[symbol] = -1/4
        
        return weights
    
    def task1_Strategy5(self, current_date_idx):
        """Strategy 5: Stochastic %K"""
        if current_date_idx < 14:
            return pd.Series(0, index=range(20))
        
        k_values = {}
        
        for symbol in range(20):
            if symbol not in self.close_prices.columns:
                k_values[symbol] = 50
                continue
                
            highs = self.high_prices.iloc[max(0, current_date_idx-14):current_date_idx][symbol].dropna()
            lows = self.low_prices.iloc[max(0, current_date_idx-14):current_date_idx][symbol].dropna()
            current_close = self.close_prices.iloc[current_date_idx-1][symbol]
            
            if len(highs) < 14 or len(lows) < 14 or pd.isna(current_close):
                k_values[symbol] = 50
                continue
            
            high_14 = highs.max()
            low_14 = lows.min()
            
            if high_14 != low_14:
                k_percent = 100 * (current_close - low_14) / (high_14 - low_14)
                k_values[symbol] = k_percent
            else:
                k_values[symbol] = 50
        
        k_series = pd.Series(k_values)
        ranked_stocks = k_series.sort_values(ascending=False)
        
        weights = pd.Series(0.0, index=range(20))
        
        # Top 3 %K get negative weights
        top_3 = ranked_stocks.head(3).index
        weights[top_3] = -1/3
        
        # Bottom 3 %K get positive weights
        bottom_3 = ranked_stocks.tail(3).index
        weights[bottom_3] = 1/3
        
        return weights

def task1_Strategy1():
    crossval_data = pd.read_csv('cross_val_data.csv')
    crossval_data = crossval_data.iloc[:, 1:]
    
    strategies = TradingStrategies(crossval_data)
    strategies.prepare_data()
    
    num_rows = len(strategies.close_prices)
    output_df = pd.DataFrame(index=range(num_rows), columns=range(20))
    output_df = output_df.fillna(0.0)
    
    for date_idx in range(250, num_rows):
        weights = strategies.task1_Strategy1(date_idx)
        output_df.loc[date_idx] = weights.values
    
    return output_df

def task1_Strategy2():
    crossval_data = pd.read_csv('cross_val_data.csv')
    crossval_data = crossval_data.iloc[:, 1:]
    
    strategies = TradingStrategies(crossval_data)
    strategies.prepare_data()
    
    num_rows = len(strategies.close_prices)
    output_df = pd.DataFrame(index=range(num_rows), columns=range(20))
    output_df = output_df.fillna(0.0)
    
    for date_idx in range(30, num_rows):
        weights = strategies.task1_Strategy2(date_idx)
        output_df.loc[date_idx] = weights.values
    
    return output_df

def task1_Strategy3():
    crossval_data = pd.read_csv('cross_val_data.csv')
    crossval_data = crossval_data.iloc[:, 1:]
    
    strategies = TradingStrategies(crossval_data)
    strategies.prepare_data()
    
    num_rows = len(strategies.close_prices)
    output_df = pd.DataFrame(index=range(num_rows), columns=range(20))
    output_df = output_df.fillna(0.0)
    
    for date_idx in range(7, num_rows):
        weights = strategies.task1_Strategy3(date_idx)
        output_df.loc[date_idx] = weights.values
    
    return output_df

def task1_Strategy4():
    crossval_data = pd.read_csv('cross_val_data.csv')
    crossval_data = crossval_data.iloc[:, 1:]
    
    strategies = TradingStrategies(crossval_data)
    strategies.prepare_data()
    
    num_rows = len(strategies.close_prices)
    output_df = pd.DataFrame(index=range(num_rows), columns=range(20))
    output_df = output_df.fillna(0.0)
    
    for date_idx in range(21, num_rows):
        weights = strategies.task1_Strategy4(date_idx)
        output_df.loc[date_idx] = weights.values
    
    return output_df

def task1_Strategy5():
    crossval_data = pd.read_csv('cross_val_data.csv')
    crossval_data = crossval_data.iloc[:, 1:]
    
    strategies = TradingStrategies(crossval_data)
    strategies.prepare_data()
    
    num_rows = len(strategies.close_prices)
    output_df = pd.DataFrame(index=range(num_rows), columns=range(20))
    output_df = output_df.fillna(0.0)
    
    for date_idx in range(14, num_rows):
        weights = strategies.task1_Strategy5(date_idx)
        output_df.loc[date_idx] = weights.values
    
    return output_df

def task1():
    Strategy1 = task1_Strategy1()
    Strategy2 = task1_Strategy2()
    Strategy3 = task1_Strategy3()
    Strategy4 = task1_Strategy4()
    Strategy5 = task1_Strategy5()

    performanceStrategy1 = backtester_without_TC(Strategy1)
    performanceStrategy2 = backtester_without_TC(Strategy2)
    performanceStrategy3 = backtester_without_TC(Strategy3)
    performanceStrategy4 = backtester_without_TC(Strategy4)
    performanceStrategy5 = backtester_without_TC(Strategy5)

    output_df = pd.DataFrame({
        'Strategy1': performanceStrategy1, 
        'Strategy2': performanceStrategy2, 
        'Strategy3': performanceStrategy3, 
        'Strategy4': performanceStrategy4, 
        'Strategy5': performanceStrategy5
    })
    output_df.to_csv('task1.csv')
    return

# ADVANCED DDQN IMPLEMENTATION WITH TRAINING DATA
class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity=20000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        """Store experience with maximum priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self.max_priority
        
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """Sample batch with prioritized replay"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class DuelingDQN(nn.Module):
    """Dueling Double Deep Q-Network"""
    
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DuelingDQN, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, action_size)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.feature_layer(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values

class AdvancedDDQNAgent:
    
    def __init__(self, state_size, action_size, lr=0.0005, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05,
                 buffer_size=20000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Neural networks
        self.q_network = DuelingDQN(state_size, action_size).to(self.device)
        self.target_network = DuelingDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=1e-4)
        
        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(buffer_size)
        
        # Learning scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=15, verbose=False
        )
        
        # Performance tracking
        self.training_rewards = []
        self.losses = []
        self.update_target_every = 100
        self.learn_every = 4
        self.step_count = 0
        
        # Initialize target network
        self.update_target_network()
    
    def update_target_network(self):
        """Soft update of target network"""
        tau = 0.005  # Soft update parameter
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in prioritized replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy with noise injection"""
        if training and np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            
            if training:
                # Add small amount of noise for exploration during training
                noise = torch.randn_like(q_values) * 0.1
                q_values += noise
        
        action = int(np.argmax(q_values.cpu().data.numpy()))
        return np.clip(action, 0, self.action_size - 1)
    
    def replay(self):
        """Train the model using prioritized experience replay"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample from prioritized replay buffer
        experiences, indices, weights = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor([e[0] for e in experiences]).to(self.device)
        actions = torch.LongTensor([e[1] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in experiences]).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute TD errors for priority update
        td_errors = (current_q_values.squeeze() - target_q_values).abs()
        
        # Weighted loss for prioritized replay
        loss = (weights_tensor * F.mse_loss(current_q_values.squeeze(), target_q_values, reduction='none')).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update priorities
        priorities = td_errors.detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, priorities)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.update_target_network()
        
        self.losses.append(loss.item())
        return loss.item()

class EnhancedMarketEnvironment:
    """Enhanced market environment for DDQN training"""
    
    def __init__(self, strategies):
        self.strategies = strategies
        self.strategy_functions = [
            strategies.task1_Strategy1,
            strategies.task1_Strategy2,
            strategies.task1_Strategy3,
            strategies.task1_Strategy4,
            strategies.task1_Strategy5
        ]
        self.state_size = 20  # Comprehensive market state
        self.action_size = 5  # Number of strategies
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.strategy_performance = {i: deque(maxlen=50) for i in range(5)}
        
    def get_market_state(self, current_idx):
        """Enhanced market state representation"""
        if current_idx < 50:
            return np.zeros(self.state_size)
        
        try:
            # Get recent price data
            lookback = min(50, current_idx)
            recent_data = self.strategies.close_prices.iloc[current_idx-lookback:current_idx]
            
            if recent_data.empty:
                return np.zeros(self.state_size)
            
            features = []
            
            # Price-based features
            returns = recent_data.pct_change().fillna(0)
            
            # 1. Average return (short and long term)
            short_returns = returns.tail(5)
            long_returns = returns.tail(20)
            features.extend([
                short_returns.mean().mean(),
                long_returns.mean().mean()
            ])
            
            # 2. Volatility (short and long term)
            features.extend([
                short_returns.std().mean(),
                long_returns.std().mean()
            ])
            
            # 3. Trend indicators
            if len(recent_data) >= 20:
                short_ma = recent_data.tail(5).mean().mean()
                long_ma = recent_data.tail(20).mean().mean()
                trend = (short_ma / long_ma - 1) if long_ma != 0 else 0
            else:
                trend = 0
            features.append(trend)
            
            # 4. Momentum indicators
            if len(recent_data) >= 10:
                momentum_5 = (recent_data.iloc[-1] / recent_data.iloc[-5] - 1).mean()
                momentum_10 = (recent_data.iloc[-1] / recent_data.iloc[-10] - 1).mean()
            else:
                momentum_5 = momentum_10 = 0
            features.extend([momentum_5, momentum_10])
            
            # 5. Cross-sectional features
            current_prices = recent_data.iloc[-1]
            cross_sect_mean = current_prices.mean()
            cross_sect_std = current_prices.std()
            features.extend([
                cross_sect_mean / recent_data.mean().mean() - 1 if recent_data.mean().mean() != 0 else 0,
                cross_sect_std / cross_sect_mean if cross_sect_mean != 0 else 0
            ])
            
            # 6. Market regime indicators
            vol_regime = 1 if returns.std().mean() > 0.02 else 0
            trend_regime = 1 if trend > 0.01 else -1 if trend < -0.01 else 0
            features.extend([vol_regime, trend_regime])
            
            # 7. Strategy-specific indicators
            for i in range(5):
                if i in self.strategy_performance and self.strategy_performance[i]:
                    avg_perf = np.mean(self.strategy_performance[i])
                    features.append(avg_perf)
                else:
                    features.append(0)
            
            # 8. Recent performance indicators
            if self.performance_history:
                recent_perf = np.mean(list(self.performance_history)[-5:])
                features.append(recent_perf)
            else:
                features.append(0)
            
            # Ensure exactly state_size features
            while len(features) < self.state_size:
                features.append(0)
            features = features[:self.state_size]
            
            # Normalize and clean
            features = np.array(features, dtype=np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            features = np.clip(features, -5, 5)  # Clip extreme values
            
            return features
            
        except Exception as e:
            print(f"Error in get_market_state: {e}")
            return np.zeros(self.state_size)
    
    def calculate_reward(self, strategy_idx, current_idx, window_size=10):
        """Enhanced reward calculation"""
        try:
            if strategy_idx >= len(self.strategy_functions):
                return 5.0  # Neutral positive reward
            
            strategy_func = self.strategy_functions[strategy_idx]
            
            # Calculate returns over window
            returns = []
            sharpe_components = []
            turnover_penalty = 0
            prev_weights = None
            
            for i in range(current_idx, min(current_idx + window_size, len(self.strategies.close_prices) - 1)):
                try:
                    weights = strategy_func(i)
                    
                    # Calculate turnover penalty
                    if prev_weights is not None:
                        turnover = abs(weights - prev_weights).sum()
                        turnover_penalty += turnover * 0.05  # Reduced penalty
                    prev_weights = weights
                    
                    # Get price returns
                    current_prices = self.strategies.close_prices.iloc[i]
                    next_prices = self.strategies.close_prices.iloc[i + 1]
                    
                    price_returns = (next_prices / current_prices - 1).fillna(0)
                    portfolio_return = (weights * price_returns).sum()
                    
                    returns.append(portfolio_return)
                    sharpe_components.append(portfolio_return)
                    
                except Exception:
                    returns.append(0)
                    sharpe_components.append(0)
            
            if len(returns) == 0:
                return 5.0
            
            total_return = sum(returns)
            avg_return = np.mean(returns)
            volatility = np.std(returns) if len(returns) > 1 else 0.001
            
            # Enhanced reward components
            return_component = total_return * 200  # Higher scaling
            sharpe_component = (avg_return / volatility) * 10 if volatility > 0 else 0
            consistency_bonus = 3.0 if all(r >= -0.001 for r in returns[-3:]) else 0  # More lenient
            turnover_penalty_scaled = turnover_penalty * 0.3  # Reduced penalty
            
            # Final reward with higher positive bias
            reward = 8.0 + return_component + sharpe_component + consistency_bonus - turnover_penalty_scaled
            
            # Update strategy performance tracking
            self.strategy_performance[strategy_idx].append(reward)
            
            # positive rewards with higher bounds
            reward = max(2.0, min(30.0, reward))
            
            return float(reward)
            
        except Exception:
            return 5.0

class DDQNStrategySelector:
    """DDQN-based strategy selector with more training data"""
    
    def __init__(self):
        self.env = None
        self.agent = None
        self.training_history = []
        
    def train(self, strategies, train_start=250, train_end=None, episodes=150):
        """Train DDQN agent with more data points"""
        print("Training Advanced DDQN Strategy Selector...")
        
        if train_end is None:
            train_end = len(strategies.close_prices) - 10  # Use more data
        
        self.env = EnhancedMarketEnvironment(strategies)
        self.agent = AdvancedDDQNAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_size,
            lr=0.0003,  
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.9996,  # Slower decay
            epsilon_min=0.05
        )
        
        training_points = list(range(train_start, train_end, 1))  # Every day!
        print(f"Training on {len(training_points)} points (EVERY DAY), {episodes} episodes")
        print(f"Total training experiences: {len(training_points) * episodes}")
        
        episode_rewards = []
        best_avg_reward = float('-inf')
        
        for episode in range(episodes):
            episode_reward = 0
            episode_loss = 0
            steps = 0
            
            points_per_episode = min(100, len(training_points)) 
            episode_points = random.sample(training_points, points_per_episode)
            
            for current_idx in episode_points:
                # Get current state
                state = self.env.get_market_state(current_idx)
                
                # Agent chooses action
                action = self.agent.act(state, training=True)
                
                # Calculate reward
                reward = self.env.calculate_reward(action, current_idx)
                
                # Get next state
                next_idx = min(current_idx + 1, train_end - 1)  # Next day
                next_state = self.env.get_market_state(next_idx)
                done = (next_idx >= train_end - 1)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train agent more frequently
                if len(self.agent.memory) > self.agent.batch_size and steps % 2 == 0:  # Every 2 steps
                    loss = self.agent.replay()
                    episode_loss += loss
                
                episode_reward += reward
                steps += 1
                
                # Update environment performance history
                self.env.performance_history.append(reward)
            
            episode_rewards.append(episode_reward / max(1, steps))
            
            # Learning rate scheduling
            if len(episode_rewards) >= 10:
                recent_avg = np.mean(episode_rewards[-10:])
                self.agent.scheduler.step(recent_avg)
                
                if recent_avg > best_avg_reward:
                    best_avg_reward = recent_avg
            
            if episode % 25 == 0:  # Every 25 episodes
                avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_rewards[-1] if episode_rewards else 0
                avg_loss = episode_loss / max(1, steps)
                print(f"  Episode {episode:3d}: Avg Reward: {avg_reward:8.3f}, "
                      f"Loss: {avg_loss:8.4f}, Epsilon: {self.agent.epsilon:.3f}, "
                      f"Buffer Size: {len(self.agent.memory)}")
                
                if hasattr(self.env, 'strategy_performance'):
                    strategy_scores = []
                    for i in range(5):
                        if self.env.strategy_performance[i]:
                            avg_score = np.mean(self.env.strategy_performance[i])
                            usage_count = len(self.env.strategy_performance[i])
                            strategy_scores.append(f"S{i+1}:{avg_score:.1f}({usage_count})")
                        else:
                            strategy_scores.append(f"S{i+1}:0.0(0)")
                    print(f"    Strategy performance: {' '.join(strategy_scores)}")
        
        self.training_history = episode_rewards
        
        print(f"\nDDQN Training Complete!")
        print(f"  Total training points used: {len(training_points)}")
        print(f"  Total experiences stored: {len(self.agent.memory)}")
        print(f"  Final epsilon: {self.agent.epsilon:.3f}")
        print(f"  Best avg reward: {best_avg_reward:.3f}")
        print(f"  Final avg reward: {np.mean(episode_rewards[-10:]):.3f}")
        
        print("\nFinal Strategy Performance:")
        for i in range(5):
            if self.env.strategy_performance[i]:
                avg_score = np.mean(self.env.strategy_performance[i])
                usage_count = len(self.env.strategy_performance[i])
                print(f"  Strategy{i+1}: {avg_score:.3f} avg score ({usage_count} uses)")
    
    def predict(self, strategies, current_idx):
        """Predict best strategy using trained DDQN"""
        if self.agent is None or self.env is None:
            return 0, 0.5
        
        # Get current market state
        state = self.env.get_market_state(current_idx)
        
        # Get Q-values for all actions
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
        with torch.no_grad():
            q_values = self.agent.q_network(state_tensor)
        
        # Select best action
        action = int(np.argmax(q_values.cpu().data.numpy()))
        
        # Calculate confidence based on Q-value spread
        q_vals = q_values.cpu().data.numpy()[0]
        max_q = np.max(q_vals)
        second_max_q = np.partition(q_vals, -2)[-2] if len(q_vals) > 1 else max_q
        confidence = min(0.95, max(0.4, (max_q - second_max_q) / (max_q + 1e-6)))
        
        return action, confidence

def task2():
    """Task 2: Advanced DDQN Strategy Selection with MORE training data"""
    print("=== TASK 2: Advanced DDQN Strategy Selection ===")
    
    # Load cross-validation data
    crossval_data = pd.read_csv('cross_val_data.csv')
    crossval_data = crossval_data.iloc[:, 1:]
    
    print(f"Cross-validation data shape: {crossval_data.shape}")
    
    # TRAINING PHASE - Use 80% for training to get MORE data
    print("\n--- TRAINING PHASE ---")
    train_size = int(len(crossval_data) * 0.8)  # 80% instead of 60%
    train_data = crossval_data.iloc[:train_size]
    
    train_strategies = TradingStrategies(train_data)
    train_strategies.prepare_data()
    
    training_data_length = len(train_strategies.close_prices)
    available_training_points = training_data_length - 250 - 10
    
    print(f"Training data size: {training_data_length} days")
    print(f"Available training points: {available_training_points} (from day 250 to {training_data_length-10})")
    
    if available_training_points < 100:
        print("WARNING: Very few training points available!")
        print("Consider using more cross-validation data or reducing train_start")
    
    ddqn_selector = DDQNStrategySelector()
    
    ddqn_selector.train(
        train_strategies,
        train_start=250,
        train_end=training_data_length - 10,
        episodes=150
    )
    
    print("\n--- TESTING PHASE ---")
    test_strategies = TradingStrategies(crossval_data)
    test_strategies.prepare_data()
    
    num_rows = len(test_strategies.close_prices)
    output_df_weights = pd.DataFrame(index=range(num_rows), columns=range(20))
    output_df_weights = output_df_weights.fillna(0.0)
    
    print("Generating DDQN predictions...")
    
    strategy_selections = []
    prediction_confidences = []
    
    strategy_functions = [
        test_strategies.task1_Strategy1,
        test_strategies.task1_Strategy2,
        test_strategies.task1_Strategy3,
        test_strategies.task1_Strategy4,
        test_strategies.task1_Strategy5
    ]
    
    for i in range(250, num_rows):
        strategy_idx, confidence = ddqn_selector.predict(test_strategies, i)
        
        weights = strategy_functions[strategy_idx](i)
        output_df_weights.loc[i] = weights.values
        strategy_selections.append(f'Strategy{strategy_idx + 1}')
        prediction_confidences.append(confidence)
        
        if (i - 250) % 50 == 0:
            print(f"  Day {i-250+1}: Strategy{strategy_idx + 1} (confidence: {confidence:.3f})")
    
    output_df_weights.to_csv('task2_weights.csv', index=False)
    results = backtester_without_TC(output_df_weights)
    
    df_performance = pd.DataFrame({
        'Net Returns': [results[0]], 
        'Sharpe Ratio': [results[1]]
    })
    df_performance.to_csv('task_2.csv', index=False)
    
    print(f"\nAdvanced DDQN Performance:")
    print(f"  Net Return: {results[0]:.4f}")
    print(f"  Sharpe Ratio: {results[1]:.4f}")
    
    print(f"\nStrategy Selection Analysis:")
    strategy_counts = pd.Series(strategy_selections).value_counts()
    for strategy, count in strategy_counts.items():
        percentage = (count / len(strategy_selections)) * 100
        print(f"  {strategy}: {count} times ({percentage:.1f}%)")
    
    avg_confidence = np.mean(prediction_confidences)
    print(f"\nAverage Prediction Confidence: {avg_confidence:.3f}")
    
    model_data = {
        'ddqn_selector': ddqn_selector,
        'method': 'Advanced DDQN',
        'performance': {'net_return': results[0], 'sharpe_ratio': results[1]},
        'training_history': ddqn_selector.training_history,
        'strategy_selections': strategy_selections,
        'prediction_confidences': prediction_confidences
    }
    
    with open('ensemble_model_ddqn.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("DDQN model saved!")
    return

def task3():
    """Task 3: Transaction cost-aware DDQN strategy selection with POSITIVE Sharpe focus"""
    print("=== TASK 3: Transaction Cost-Aware DDQN Selection ===")
    
    try:
        with open('ensemble_model_ddqn.pkl', 'rb') as f:
            model_data = pickle.load(f)
        ddqn_selector = model_data['ddqn_selector']
        print("Loaded trained DDQN model")
    except:
        print("No trained model found, training new model...")
        task2()
        with open('ensemble_model_ddqn.pkl', 'rb') as f:
            model_data = pickle.load(f)
        ddqn_selector = model_data['ddqn_selector']
    
    train_data = pd.read_csv('train_data.csv')
    train_data = train_data.iloc[:, 1:]
    
    strategies = TradingStrategies(train_data)
    strategies.prepare_data()
    
    num_rows = len(strategies.close_prices)
    output_df_weights = pd.DataFrame(index=range(num_rows), columns=range(20))
    output_df_weights = output_df_weights.fillna(0.0)
    
    prev_weights = pd.Series(0.0, index=range(20))
    
    print("Generating ULTRA CONSERVATIVE transaction cost-aware weights...")
    
    strategy_functions = [
        strategies.task1_Strategy1,
        strategies.task1_Strategy2,
        strategies.task1_Strategy3,
        strategies.task1_Strategy4,
        strategies.task1_Strategy5
    ]
    
    turnover_history = deque(maxlen=50)
    strategy_stability_tracker = deque(maxlen=20)
    cumulative_tc = 0
    strategy_switch_penalty = 0
    
    for date_idx in range(250, num_rows):
        # DDQN prediction
        strategy_idx, confidence = ddqn_selector.predict(strategies, date_idx)
        
        # CONSERVATIVE Strategy stability tracking
        strategy_stability_tracker.append(strategy_idx)
        
        # Heavily penalize strategy switching
        if len(strategy_stability_tracker) >= 3:
            recent_strategies = list(strategy_stability_tracker)[-3:]
            if len(set(recent_strategies)) > 1:  # Any switching in last 3 days
                confidence *= 0.3  # Massive confidence reduction
                strategy_switch_penalty += 0.1
        
        # If switching too much, force to stay with most recent successful strategy
        if len(strategy_stability_tracker) >= 10:
            strategy_changes = 0
            for i in range(1, len(strategy_stability_tracker)):
                if strategy_stability_tracker[i] != strategy_stability_tracker[i-1]:
                    strategy_changes += 1
            
            if strategy_changes > 3:  # More than 3 changes in 10 days
                # Force to use the most common recent strategy
                recent_10 = list(strategy_stability_tracker)[-10:]
                most_common = max(set(recent_10), key=recent_10.count)
                strategy_idx = most_common
                confidence = 0.2  # Very low confidence
        
        # Get weights from selected strategy
        new_weights = strategy_functions[strategy_idx](date_idx)
        
        # Calculate potential turnover
        weight_change = abs(new_weights - prev_weights).sum()
        turnover_history.append(weight_change)
        

        base_blend_factor = 0.25  # Increased from 0.15
                
        if confidence < 0.3:
            blend_factor = 0.15   # Increased from 0.05
        elif confidence < 0.5:
            blend_factor = 0.20   # Increased from 0.10
        elif confidence < 0.7:
            blend_factor = 0.25   # Increased from 0.15
        else:
            blend_factor = 0.35   # Increased from 0.25

        # Reduce turnover penalties:
        if weight_change > 0.5:
            blend_factor *= 0.6   
        elif weight_change > 0.3:
            blend_factor *= 0.7   
        elif weight_change > 0.2:
            blend_factor *= 0.8   

        # Reduce strategy switching penalties:
        confidence *= 0.7  # Reduced from 0.3 for strategy switching

    
                
        new_weights = blend_factor * new_weights + (1 - blend_factor) * prev_weights
        
        # Final emergency brake - ensure turnover never exceeds 0.3
        final_turnover = abs(new_weights - prev_weights).sum()
        if final_turnover > 0.3:
            emergency_blend = 0.3 / (final_turnover + 1e-6)
            emergency_blend = min(emergency_blend, 0.1)
            new_weights = emergency_blend * new_weights + (1 - emergency_blend) * prev_weights
            final_turnover = abs(new_weights - prev_weights).sum()
        
        # Track cumulative transaction costs
        cumulative_tc += final_turnover * 0.01
        
        output_df_weights.loc[date_idx] = new_weights.values
        prev_weights = new_weights.copy()
        
        if date_idx % 1000 == 0:
            recent_turnover = np.mean(list(turnover_history)[-10:]) if len(turnover_history) >= 10 else 0
            print(f"  Day {date_idx}: Strategy{strategy_idx+1}, "
                  f"Confidence: {confidence:.3f}, "
                  f"Recent turnover: {recent_turnover:.4f}, "
                  f"Cumulative TC: {cumulative_tc:.4f}")
    
    output_df_weights.to_csv('task3_weights.csv', index=False)
    results = backtester_with_TC(output_df_weights)
    
    df_performance = pd.DataFrame({
        'Net Returns': [results[0]], 
        'Sharpe Ratio': [results[1]]
    })
    df_performance.to_csv('task_3.csv', index=False)
    
    print(f"\nTask 3 ULTRA CONSERVATIVE DDQN Performance:")
    print(f"  Net Return: {results[0]:.4f}")
    print(f"  Sharpe Ratio: {results[1]:.4f}")
    
    weights_df = output_df_weights.fillna(0)
    daily_turnover = abs(weights_df.diff()).sum(axis=1)
    daily_turnover = daily_turnover.fillna(0)
    
    avg_daily_turnover = daily_turnover.mean()
    max_daily_turnover = daily_turnover.max()
    total_turnover = daily_turnover.sum()
    total_tc_cost = total_turnover * 0.01
    
    print(f"\nULTRA CONSERVATIVE Transaction Cost Analysis:")
    print(f"  Average Daily Turnover: {avg_daily_turnover:.6f}")
    print(f"  Maximum Daily Turnover: {max_daily_turnover:.6f}")
    print(f"  Total Turnover: {total_turnover:.4f}")
    print(f"  Total Transaction Costs: {total_tc_cost:.6f}")
    print(f"  Strategy Switch Penalty: {strategy_switch_penalty:.4f}")
    
    zero_turnover_days = (daily_turnover == 0).sum()
    print(f"  Days with zero turnover: {zero_turnover_days} ({zero_turnover_days/len(daily_turnover)*100:.1f}%)")
    
    if len(strategy_stability_tracker) > 1:
        strategy_changes = pd.Series([s for s in strategy_stability_tracker]).diff().fillna(0)
        switch_rate = (strategy_changes != 0).sum() / len(strategy_changes)
        print(f"  Strategy Switch Rate: {switch_rate:.3%}")
    
    return

if __name__ == '__main__':
    task1()
    task2()
    task3()