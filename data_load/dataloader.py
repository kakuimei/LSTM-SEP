from summarize_module.summarizer import DeepSeekSummarizer
import os, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from memory_module.memorydb import BrainDB
import calendar


class DataLoader:
    def __init__(self, args):
        self.price_dir = args.price_dir
        self.tweet_dir = args.tweet_dir
        self.seq_len = args.seq_len
        self.summarizer = DeepSeekSummarizer()
        self.brain_db = BrainDB()

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
            with open(tweet_path) as f:
                lines = f.readlines()
                for line in lines:
                    tweet_obj = json.loads(line)
                    tweets.append(tweet_obj['text'])
        return tweets


    def load(self, flag):

        for file in os.listdir(self.price_dir):
            price_path = os.path.join(self.price_dir, file)
            ordered_price_data = np.flip(np.genfromtxt(price_path, dtype=str, skip_header=False), 0)
            ticker = file[:-4]

            # Get the date range
            date_to_row = {
                datetime.strptime(row[0], "%Y-%m-%d").date(): row
                for row in ordered_price_data
            }
            available_dates = set(date_to_row.keys())
            split_idx = round(len(ordered_price_data) * 0.8)
            split_date = datetime.strptime(ordered_price_data[split_idx, 0], "%Y-%m-%d").date()
            min_date = min(available_dates)
            max_date = max(available_dates)
            full_dates = []
            cur = min_date
            while cur <= max_date:
                full_dates.append(cur)
                cur += timedelta(days=1)
            if flag == "train":
                data_range = [d for d in full_dates if d < split_date]
            else:
                data_range = [d for d in full_dates if d >= split_date]
            
            weekly_summary_dict = {} 
            monthly_summary_dict = {}
            for date in data_range:
                # get tweet data
                date_str = date.strftime("%Y-%m-%d")
                summary_all_parts = []
                tweet_data = self.get_tweets(ticker, date_str)
                if tweet_data:
                    
                    # daily_summary
                    daily_summary = self.summarizer.get_summary(ticker, tweet_data)
                    if daily_summary and self.summarizer.is_informative(daily_summary):
                        self.brain_db.add_memory_short(symbol=ticker,date=date,text=daily_summary)
                        summary_all_parts.append(f"{date_str} Daily: {daily_summary}")
                        weekly_summary_dict[date_str] = daily_summary
                    else:
                        weekly_summary_dict[date_str] = "[Uninformative tweet summary]"
                else:
                    weekly_summary_dict[date_str] = "[No tweets available]"
                
                # weekly_summary
                if date.weekday() == 6:
                    week_start = date - timedelta(days=6)
                    week_dates = [(week_start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
                    daily_lines = [f"{d}: {weekly_summary_dict.get(d)}" for d in week_dates]
                    raw_weekly_report = "\n".join(daily_lines)
                    weekly_summary = self.summarizer.get_weekly_summary(
                        ticker=ticker, week_start=week_start, week_end= date, raw_weekly_report=raw_weekly_report
                        )
                    self.brain_db.add_memory_mid(symbol=ticker, date=date, text=weekly_summary)
                    monthly_summary_dict[date.strftime("%Y-%m-%d")] = weekly_summary
                    summary_all_parts.append(f"{date_str} Weekly: {weekly_summary}")
                    weekly_summary_dict.clear()
                
                # monthly_summary
                last_day = calendar.monthrange(date.year, date.month)[1]
                if date.day == last_day:
                    month_start = date.replace(day=1)
                    month_end = date
                    month_weeks = sorted(monthly_summary_dict.keys())
                    raw_monthly_report = "\n".join(
                        f"Week ending {wd}: {monthly_summary_dict[wd]}" for wd in month_weeks
                        )
                    monthly_summary = self.summarizer.get_monthly_summary(
                        ticker=ticker, month_start=month_start, month_end=month_end, raw_monthly_report=raw_monthly_report
                        )
                    self.brain_db.add_memory_long(symbol=ticker, date=date, text=monthly_summary)
                    summary_all_parts.append(f"{date_str} Monthly: {monthly_summary}")
                    monthly_summary_dict.clear()
                
                # get the target sentiment
                if date in available_dates:
                    target = self.get_sentiment(date_str, price_path)
                else:
                    target = None
            
                # if daily_summary and daily_summary is not None and daily_summary != "" and self.summarizer.is_informative(summary):
                #     summary_all = summary_all + date_str + "\n" + daily_summary + "\n\n"

                if summary_all_parts:
                    yield {
                        'ticker': ticker,
                        'date': date,
                        'summary': "\n\n".join(summary_all_parts),
                        'target': target
                    }
