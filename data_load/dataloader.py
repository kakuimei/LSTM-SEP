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
        self.ckpt_dir = args.ckpt_dir
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
            print(f"[Info] Loading price data for {ticker}, {len(ordered_price_data)} days from {ordered_price_data[0,0]} to {ordered_price_data[-1,0]}")
            
            date_to_row = {datetime.strptime(row[0], "%Y-%m-%d").date(): row for row in ordered_price_data}
            split_idx = round(len(ordered_price_data) * 0.8)
            split_date = datetime.strptime(ordered_price_data[split_idx, 0], "%Y-%m-%d").date()
            min_d, max_d = min(date_to_row), max(date_to_row)
            full_dates = [min_d + timedelta(days=i) for i in range((max_d - min_d).days + 1)]
            keep = (lambda d: d < split_date) if flag == "train" else (lambda d: d >= split_date)
            data_range = [d for d in full_dates if keep(d)]

            # To avoid re-processing data, find the latest date we have done for this ticker
            latest_done_date = None
            for mem in (self.brain_db.short_term_memory,
                        self.brain_db.mid_term_memory,
                        self.brain_db.long_term_memory,
                        self.brain_db.reflection_memory):
                if (ticker in mem.universe) and (len(mem.universe[ticker]["score_memory"]) > 0):
                    for rec in mem.universe[ticker]["score_memory"]:
                        d = rec.get("date", None)
                        if d is not None:
                            if isinstance(d, str):
                                d = datetime.strptime(d, "%Y-%m-%d").date()
                            if latest_done_date is None or d > latest_done_date:
                                latest_done_date = d

            if latest_done_date is None:
                latest_done_date = min_d - timedelta(days=1)
            last_saved_date = latest_done_date
            print(f"[Info] Will skip dates <= {latest_done_date} for {ticker}")
            
            data_range = [d for d in data_range if d > latest_done_date]
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
                        print(f"[Info] Added daily summary to short memory for {ticker} on {date_str}")
                        daily_summary_dict[date_str] = daily_summary
                    else:
                        daily_summary_dict[date_str] = "[Uninformative tweet summary]"
                        print(f"[Info] Uninformative daily summary for {ticker} on {date_str}")
                    short_ctx, _ = self.brain_db.query_short(query_text, 5, ticker)
                    summary_all_parts.append(f"{date_str} Daily: \n{short_ctx}\n\n")
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
                        print(f"[Info] Added weekly summary to mid memory for {ticker} on {week_end_str}")
                        weekly_summary_dict[week_end_str] = weekly_summary
                    else:
                        weekly_summary_dict[week_end_str] = "[Uninformative weekly summary]"
                        print(f"[Info] Uninformative weekly summary for {ticker} on {week_end_str}")
                    mid_ctx,   _ = self.brain_db.query_mid(query_text, 1, ticker)
                    summary_all_parts.append(f"Week {week_start_str} to {week_end_str} Summary: \n{mid_ctx}\n\n")
                
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
                    print(f"[Info] Added monthly summary to long memory for {ticker} on {month_end_str}")
                    long_ctx,  _ = self.brain_db.query_long(query_text, 1, ticker)
                    summary_all_parts.append(f"Month {month_start_str} to {month_end_str} Summary: \n{long_ctx}\n\n")
                
                # get the target sentiment
                if date in date_to_row:
                    target = self.get_sentiment(date_str, price_path)
                else:
                    target = None
                # Step BrainDB
                if (date - min_d).days % 7 == 0:
                    self.brain_db.step()
                # Save checkpoint every 7 days
                if (date - last_saved_date).days >= 7:
                    os.makedirs(self.ckpt_dir, exist_ok=True)
                    print(f"[Info] 7-day progress reached at {ticker} {date_str}. Saving BrainDB...")
                    self.brain_db.save_checkpoint(self.ckpt_dir, force=True)
                    last_saved_date = date
            

                if summary_all_parts and target is not None:
                    yield {
                        'ticker': ticker,
                        'date': date_str,
                        'summary': "\n\n".join(summary_all_parts),
                        'target': target
                    }
