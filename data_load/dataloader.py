import os, json
import numpy as np
from datetime import datetime, timedelta
import logging
from memory_module.memorydb import BrainDB
from summarize_module.summarizer import DeepSeekSummarizer

class DataLoader:
    def __init__(self, args, brain_db=None, summarizer=None):
        self.price_dir = args.price_dir
        self.tweet_dir = args.tweet_dir
        self.seq_len = args.seq_len
        self.summarizer = summarizer if summarizer else DeepSeekSummarizer()
        self.brain_db = brain_db if brain_db else BrainDB.from_config(args)

    def daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def get_sentiment(self, date_str, price_path):
        price_data = np.genfromtxt(price_path, dtype=str, skip_header=False)
        price_chg = price_data[price_data[:, 0] == date_str][0, 1].astype(float)

        if price_chg > 0.0:
            sentiment = "Positive"
        else:
            sentiment = "Negative"
        return sentiment


    def get_tweets(self, ticker, date_str):
        tweets = []
        tweet_path = os.path.join(self.tweet_dir, ticker, date_str)
        if os.path.exists(tweet_path):
            with open(tweet_path, encoding="utf-8") as f:
                for line in f:
                    tweet_obj = json.loads(line)
                    text = tweet_obj.get("text", "")
                    if isinstance(text, list):
                        text = " ".join(map(str, text))
                    text = text.strip()
                    if text:
                        tweets.append(text)
        else:
            print(f"[Info] No tweets found for {ticker} on {date_str}")
        return "\n".join(tweets) if tweets else ""


    def load(self, flag):
        for file in os.listdir(self.price_dir):
            price_path = os.path.join(self.price_dir, file)
            ordered_price_data = np.flip(np.genfromtxt(price_path, dtype=str, skip_header=False), 0)
            ticker = file[:-4]

            # set up logging for this ticker
            log_filename = self.brain_db.logger.log_filename_template.format(symbol=ticker)
            log_path = os.path.join(self.brain_db.logger.log_dir, log_filename)

            # ↓↓↓ 只在还没加过这个文件 handler 的时候再添加（关键一行） ↓↓↓
            already = any(isinstance(h, logging.FileHandler) and getattr(h, "_log_path", None) == log_path
                        for h in self.brain_db.logger.handlers)
            if not already:
                file_handler = logging.FileHandler(log_path, mode="a")
                file_handler._log_path = log_path  # 做个标记，便于上面的判断
                file_handler.setFormatter(logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                ))
                self.brain_db.logger.addHandler(file_handler)
            print(f"[Info] Loading price data for {ticker}, {len(ordered_price_data)} days from {ordered_price_data[0,0]} to {ordered_price_data[-1,0]}")
            
            date_to_row = {datetime.strptime(row[0], "%Y-%m-%d").date(): row for row in ordered_price_data}
            split_idx = round(len(ordered_price_data) * 0.8)
            split_date = datetime.strptime(ordered_price_data[split_idx, 0], "%Y-%m-%d").date()
            min_d, max_d = min(date_to_row), max(date_to_row)
            full_dates = [min_d + timedelta(days=i) for i in range((max_d - min_d).days + 1)]
            keep = (lambda d: d < split_date) if flag == "train" else (lambda d: d >= split_date)
            data_range = [d for d in full_dates if keep(d)]

            print(f"[Info] Processing train data for {ticker}, {len(data_range)} days from {data_range[0]} to {data_range[-1]}....")
            daily_summary_dict = {}
            weekly_summary_dict = {}
            for date in data_range:
                summary_all_parts = []
                # pull tweets for the day
                date_str = date.strftime("%Y-%m-%d")
                tweet_data = self.get_tweets(ticker, date_str)
                query_text = self.summarizer.get_query_text(ticker, date_str, tweet_data)

                if tweet_data:
                    # write daily_summary to short memory
                    daily_summary = self.summarizer.get_summary(ticker, tweet_data)
                    if daily_summary and self.summarizer.is_informative(daily_summary):
                        self.brain_db.add_memory_short(symbol=ticker,date=date,text=daily_summary)
                        print(f"[Info] Added daily summary to mid memory for {ticker} on {date_str}")
                        daily_summary_dict[date_str] = daily_summary
                    else:
                        daily_summary_dict[date_str] = "[Uninformative tweet summary]"
                        print(f"[Info] Uninformative daily summary for {ticker} on {date_str}")
                    summary_all_parts.append(f"{date_str} Daily: \n{daily_summary}\n\n")
                else:
                    daily_summary_dict[date_str] = "[No tweets available]"
                    print(f"[Info] No tweets available for {ticker} on {date_str}")
                
                # write weekly_summary to long memory
                if date >= min_d + timedelta(days=6):
                    week_start = date - timedelta(days=6)
                    week_dates = [(week_start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
                    week_start_str = week_start.strftime("%Y-%m-%d")
                    week_end_str = date.strftime("%Y-%m-%d")
                    weekly_lines = [f"{d}: {daily_summary_dict.get(d,'[No data]')}" for d in week_dates]
                    raw_weekly = "\n".join(weekly_lines)
                    weekly_summary = self.summarizer.get_weekly_summary(
                        ticker=ticker, week_start=week_start_str, week_end=week_end_str, raw_weekly_report=raw_weekly
                        )
                    if weekly_summary and self.summarizer.is_informative(weekly_summary):
                        self.brain_db.add_memory_mid(symbol=ticker, date=week_end_str, text=weekly_summary)
                        print(f"[Info] Added weekly summary to long memory for {ticker} on {week_end_str}")
                        weekly_summary_dict[week_end_str] = weekly_summary
                    else:
                        weekly_summary_dict[week_end_str] = "[Uninformative weekly summary]"
                        print(f"[Info] Uninformative weekly summary for {ticker} on {week_end_str}")
                    summary_all_parts.append(f"Week {week_start_str} to {week_end_str} Summary: \n{weekly_summary}\n\n")
                
                # write monthly_summary to long memory
                if date >= min_d + timedelta(days=29):
                    month_start = date - timedelta(days=29)
                    month_start_str = month_start.strftime("%Y-%m-%d")
                    month_end_str = date.strftime("%Y-%m-%d")
                    monthly_lines = []
                    for i in range(0, 22, 7):
                        week_end_date = (date - timedelta(days=i)).strftime("%Y-%m-%d")
                        summary_text = weekly_summary_dict.get(week_end_date, "[No weekly summary]")
                        monthly_lines.append(f"Week of {week_end_date}: {summary_text}")
                    raw_monthly = "\n".join(monthly_lines)
                    monthly_summary = self.summarizer.get_monthly_summary(
                        ticker=ticker,
                        month_start=month_start_str,
                        month_end=month_end_str,
                        raw_monthly_report=raw_monthly,
                    )
                    self.brain_db.add_memory_long(symbol=ticker, date=month_end_str, text=monthly_summary)
                    summary_all_parts.append(f"Month {month_start_str} to {month_end_str} Summary: \n{monthly_summary}\n\n")
                    print(f"[Info] Added monthly summary to long memory for {ticker} on {month_end_str}")
                
                # get the target sentiment
                if date in date_to_row:
                    target = self.get_sentiment(date_str, price_path)
                else:
                    target = None

                if (date - min_d).days % 7 == 0:
                    self.brain_db.step()

                long_ctx,  _ = self.brain_db.query_long(query_text, 2, ticker)
                mid_ctx,   _ = self.brain_db.query_mid(query_text, 3, ticker)
                short_ctx, _ = self.brain_db.query_short(query_text, 5, ticker)

                if summary_all_parts:
                    yield {
                        'ticker': ticker,
                        'date': date_str,
                        'summary': "\n\n".join(summary_all_parts),
                        'target': target,
                        'ctx_short': short_ctx,
                        'ctx_mid':   mid_ctx,
                        'ctx_long':  long_ctx,
                    }
