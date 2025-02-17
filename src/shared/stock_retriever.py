from typing import List, Dict, Optional
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from nano_vectordb import NanoVectorDB
from langchain_openai import OpenAIEmbeddings

class StockRetriever:
    """Retriever for stock market data and analysis"""
    
    def __init__(self):
        self.cache = {}  # Simple cache for stock data
        self.embeddings = OpenAIEmbeddings()
        self.news_store = None
        self.vector_dim = 1536  # OpenAI embedding dimension
        
    def get_stock_data(self, symbol: str, period: str = "1y") -> Dict:
        """Get basic stock data and technical indicators"""
        if not symbol:
            return {"error": "No stock symbol provided"}
            
        # Check cache first
        cache_key = f"{symbol}_{period}"
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if datetime.now() - cache_time < timedelta(minutes=15):  # 15-min cache
                return data
        
        try:
            stock = yf.Ticker(symbol)
            # Get historical data
            hist = stock.history(period=period)
            
            if hist.empty:
                return {"error": f"No data found for {symbol}"}
                
            # Calculate technical indicators
            hist['SMA20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA50'] = hist['Close'].rolling(window=50).mean()
            hist['RSI'] = self._calculate_rsi(hist['Close'])
            hist['MACD'] = self._calculate_macd(hist['Close'])
            hist['BB_upper'], hist['BB_middle'], hist['BB_lower'] = self._calculate_bollinger_bands(hist['Close'])
            
            # Get latest data point
            latest = hist.iloc[-1]
            prev_close = hist.iloc[-2]['Close'] if len(hist) > 1 else latest['Close']
            
            # Calculate daily change
            daily_change = ((latest['Close'] - prev_close) / prev_close * 100) if prev_close != 0 else 0
            
            # Get 52-week range
            year_high = hist['High'].max()
            year_low = hist['Low'].min()
            
            # Get company info
            info = self._get_company_info(stock)
            
            # Prepare result
            result = {
                "symbol": symbol,
                "current_price": float(latest['Close']),
                "daily_change": float(daily_change),
                "volume": int(latest['Volume']),
                "sma20": float(latest['SMA20']) if not pd.isna(latest['SMA20']) else None,
                "sma50": float(latest['SMA50']) if not pd.isna(latest['SMA50']) else None,
                "rsi": float(latest['RSI']) if not pd.isna(latest['RSI']) else None,
                "macd": float(latest['MACD']) if not pd.isna(latest['MACD']) else None,
                "bollinger_bands": {
                    "upper": float(latest['BB_upper']) if not pd.isna(latest['BB_upper']) else None,
                    "middle": float(latest['BB_middle']) if not pd.isna(latest['BB_middle']) else None,
                    "lower": float(latest['BB_lower']) if not pd.isna(latest['BB_lower']) else None
                },
                "high_52w": float(year_high),
                "low_52w": float(year_low),
                "company_info": info,
                "technical_signals": self._analyze_technical_signals(latest)
            }
            
            # Cache the result
            self.cache[cache_key] = (datetime.now(), result)
            return result
            
        except Exception as e:
            error_msg = str(e)
            if "Invalid API call" in error_msg:
                return {"error": f"Invalid stock symbol: {symbol}"}
            elif "failed to download" in error_msg:
                return {"error": f"Failed to download data for {symbol}. Please check your internet connection."}
            else:
                return {"error": f"Unexpected error for {symbol}: {error_msg}"}
    
    def get_market_news(self, symbol: str) -> List[Dict]:
        """Get recent news about the stock"""
        try:
            stock = yf.Ticker(symbol)
            news = stock.news or []
            
            formatted_news = []
            for item in news[:20]:  # Get 20 most recent news items
                if not isinstance(item, dict):
                    continue
                
                try:
                    # Extract content from the nested structure
                    content = item.get('content', {}) if isinstance(item.get('content'), dict) else item
                    
                    # Get provider info
                    provider = content.get('provider', {})
                    provider_name = provider.get('displayName', 'Unknown')
                    
                    # Get URL from canonical or clickThrough
                    url_info = content.get('canonicalUrl', content.get('clickThroughUrl', {}))
                    link = url_info.get('url', '#')
                    
                    # Get publication date
                    pub_date = content.get('pubDate', content.get('displayTime', ''))
                    if pub_date:
                        try:
                            # Convert timestamp if it's a number
                            if isinstance(pub_date, (int, float)):
                                pub_date = datetime.fromtimestamp(pub_date).strftime("%Y-%m-%d %H:%M:%S")
                        except (ValueError, TypeError):
                            pub_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        pub_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Get summary if available
                    summary = content.get('summary', content.get('description', ''))
                    
                    # Categorize news based on keywords
                    title = str(content.get("title", ""))
                    category = self._categorize_news(title, summary)
                    
                    news_item = {
                        "title": title,
                        "summary": str(summary),
                        "publisher": str(provider_name),
                        "link": str(link),
                        "published": pub_date,
                        "category": category,
                        "embedding": self.embeddings.embed_query(title + " " + summary) if self.embeddings else None
                    }
                    
                    # Only add if we have a title
                    if news_item["title"].strip():
                        formatted_news.append(news_item)
                        
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Error processing news item: {e}")
                    continue
            
            return formatted_news if formatted_news else [
                {
                    "title": "No recent news available",
                    "publisher": "System",
                    "link": "#",
                    "published": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "category": "system",
                    "summary": "",
                    "embedding": None
                }
            ]
        except Exception as e:
            print(f"Error fetching news: {e}")
            return [{
                "title": "Unable to fetch news",
                "publisher": "System",
                "link": "#",
                "published": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "category": "system",
                "summary": "",
                "embedding": None
            }]
    
    def search_related_news(self, query: str, k: int = 5) -> List[Dict]:
        """Search for related news articles using semantic search"""
        if not self.news_store:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in nano-vectordb
            results = self.news_store.query(
                np.array(query_embedding), 
                top_k=k,
                better_than_threshold=0.5  # Adjust threshold as needed
            )
            
            return [item for item in results if item]
        except Exception as e:
            print(f"Error searching news: {e}")
            return []
    
    def get_news_by_category(self, category: str) -> List[Dict]:
        """Get all news items for a specific category"""
        if not self.news_store:
            return []
        
        try:
            # Use semantic search with category description
            category_query = f"News articles about {category} related topics"
            query_embedding = self.embeddings.embed_query(category_query)
            
            # Search with filter for exact category match
            results = self.news_store.query(
                np.array(query_embedding),
                top_k=20,  # Get up to 20 results per category
                filter_lambda=lambda x: x.get("category") == category
            )
            
            return [item for item in results if item]
        except Exception as e:
            print(f"Error getting news by category: {e}")
            return []
            
    def _create_news_store(self, news_items: List[Dict]):
        """Create a vector store from news items"""
        try:
            # Initialize nano-vectordb
            self.news_store = NanoVectorDB(self.vector_dim)
            
            # Prepare data for vector store
            for item in news_items:
                # Combine title and summary for better semantic search
                content = f"{item['title']}\n{item['summary']}"
                
                # Get embedding
                embedding = self.embeddings.embed_query(content)
                
                # Add to vector store
                self.news_store.upsert([{
                    "__vector__": np.array(embedding),
                    **item  # Include all metadata
                }])
                
        except Exception as e:
            print(f"Error creating news store: {e}")
            self.news_store = None
            
    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(0)  # Replace NaN with 0
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return pd.Series([0] * len(prices))  # Return zeros on error
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate Moving Average Convergence Divergence"""
        try:
            exp1 = prices.ewm(span=fast, adjust=False).mean()
            exp2 = prices.ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            return macd - signal_line  # Return MACD histogram
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            return pd.Series([0] * len(prices))

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
        """Calculate Bollinger Bands"""
        try:
            middle = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            upper = middle + (std * num_std)
            lower = middle - (std * num_std)
            return upper, middle, lower
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            zeros = pd.Series([0] * len(prices))
            return zeros, zeros, zeros

    def _analyze_technical_signals(self, latest_data: pd.Series) -> Dict[str, str]:
        """Analyze technical indicators for trading signals"""
        signals = {}
        
        # RSI Analysis
        if not pd.isna(latest_data.get('RSI')):
            rsi = latest_data['RSI']
            if rsi > 70:
                signals['rsi'] = 'overbought'
            elif rsi < 30:
                signals['rsi'] = 'oversold'
            else:
                signals['rsi'] = 'neutral'
        
        # MACD Analysis
        if not pd.isna(latest_data.get('MACD')):
            macd = latest_data['MACD']
            if macd > 0:
                signals['macd'] = 'bullish'
            else:
                signals['macd'] = 'bearish'
        
        # Bollinger Bands Analysis
        if all(not pd.isna(latest_data.get(x)) for x in ['BB_upper', 'BB_middle', 'BB_lower', 'Close']):
            close = latest_data['Close']
            upper = latest_data['BB_upper']
            lower = latest_data['BB_lower']
            
            if close > upper:
                signals['bollinger'] = 'overbought'
            elif close < lower:
                signals['bollinger'] = 'oversold'
            else:
                signals['bollinger'] = 'neutral'
        
        # Moving Average Analysis
        if all(not pd.isna(latest_data.get(x)) for x in ['SMA20', 'SMA50', 'Close']):
            close = latest_data['Close']
            sma20 = latest_data['SMA20']
            sma50 = latest_data['SMA50']
            
            if close > sma20 and sma20 > sma50:
                signals['moving_averages'] = 'strong_bullish'
            elif close > sma20:
                signals['moving_averages'] = 'bullish'
            elif close < sma20 and sma20 < sma50:
                signals['moving_averages'] = 'strong_bearish'
            elif close < sma20:
                signals['moving_averages'] = 'bearish'
            else:
                signals['moving_averages'] = 'neutral'
                
        return signals

    def _get_company_info(self, stock: yf.Ticker) -> Dict:
        """Get basic company information"""
        try:
            info = stock.info or {}
            return {
                "name": str(info.get("longName", info.get("shortName", ""))),
                "sector": str(info.get("sector", "")),
                "industry": str(info.get("industry", "")),
                "market_cap": int(info.get("marketCap", 0)),
                "pe_ratio": float(info.get("trailingPE", 0)),
                "dividend_yield": float(info.get("dividendYield", 0) or 0),
            }
        except Exception as e:
            print(f"Error getting company info: {e}")
            return {
                "name": "",
                "sector": "",
                "industry": "",
                "market_cap": 0,
                "pe_ratio": 0,
                "dividend_yield": 0,
            }
            
    def _categorize_news(self, title: str, summary: str) -> str:
        """Categorize news based on content with improved categories"""
        text = (title + " " + summary).lower()
        
        categories = {
            "earnings": [
                "earnings", "revenue", "profit", "loss", "eps", "quarter", 
                "financial results", "guidance", "forecast", "outlook"
            ],
            "product": [
                "launch", "product", "release", "announced", "unveils",
                "innovation", "development", "patent", "research"
            ],
            "management": [
                "ceo", "executive", "management", "appointed", "resigned",
                "board", "director", "leadership", "strategy"
            ],
            "market": [
                "market", "stock", "shares", "trading", "investors",
                "valuation", "analyst", "rating", "upgrade", "downgrade"
            ],
            "regulatory": [
                "sec", "regulation", "compliance", "legal", "lawsuit",
                "investigation", "settlement", "fine", "approval"
            ],
            "partnership": [
                "partnership", "collaboration", "deal", "agreement",
                "merger", "acquisition", "joint venture", "alliance"
            ],
            "industry": [
                "industry", "sector", "competition", "market share",
                "trend", "disruption", "growth", "decline"
            ],
            "financial": [
                "dividend", "debt", "financing", "investment", "acquisition",
                "restructuring", "cost", "expense", "capital"
            ],
            "risk": [
                "risk", "warning", "concern", "issue", "problem",
                "challenge", "threat", "uncertainty", "volatility"
            ],
            "technology": [
                "technology", "digital", "software", "platform", "cloud",
                "ai", "automation", "cybersecurity", "data"
            ]
        }
        
        # Score-based categorization
        category_scores = {cat: 0 for cat in categories}
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in text:
                    category_scores[category] += text.count(keyword)
        
        # Get category with highest score
        max_score = max(category_scores.values())
        if max_score > 0:
            return max(category_scores.items(), key=lambda x: x[1])[0]
                
        return "general"  # Default category
