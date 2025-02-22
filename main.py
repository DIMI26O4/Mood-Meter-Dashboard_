import pandas as pd
import plotly.express as px
from textblob import TextBlob

# Load the dataset
df = pd.read_csv('C:/Users/my lapi/Desktop/DV_21_02_25/DV_21_02_25/sleep_cycle_productivity.csv')
#to check for non null values (the data is already clean)
df.info()
# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Perform sentiment analysis on the 'Mood Score' column
df['Sentiment'] = df['Mood Score'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Correlation analysis
correlation_matrix = df[['Sleep Quality', 'Productivity Score', 'Total Sleep Hours', 'Mood Score', 'Exercise (mins/day)', 'Stress Level', 'Screen Time Before Bed (mins)', 'Sentiment']].corr()

# heatmap of the correlation matrix
fig_heatmap = px.imshow(correlation_matrix, text_auto=True, aspect="auto")
fig_heatmap.update_layout(title="Correlation Heatmap")
fig_heatmap.show()

# scatter plot of Sleep Quality vs Productivity Score
fig_scatter = px.scatter(df, x="Sleep Quality", y="Productivity Score", color="Sentiment",
                         hover_data=["Total Sleep Hours", "Exercise (mins/day)"])
fig_scatter.update_layout(title="Sleep Quality vs Productivity Score")
fig_scatter.show()

# box plot of Stress Level for different Exercise durations
df['Exercise_Category'] = pd.cut(df['Exercise (mins/day)'], bins=[0, 30, 60, 90, 120], labels=['0-30', '31-60', '61-90', '91+'])
fig_box = px.box(df, x="Exercise_Category", y="Stress Level")
fig_box.update_layout(title="Stress Level vs Exercise Duration")
fig_box.show()

# Data filtering and analysis
high_productivity = df[df['Productivity Score'] > 8]
low_stress = df[df['Stress Level'] < 4]
optimal_sleep = df[(df['Total Sleep Hours'] >= 7) & (df['Total Sleep Hours'] <= 9)]

# Create bar plots for the filtered data
fig_high_prod = px.bar(high_productivity[['Sleep Quality', 'Exercise (mins/day)', 'Caffeine Intake (mg)']].mean().reset_index(),
                       x='index', y=0, title="High Productivity Group")
fig_high_prod.show()

fig_low_stress = px.bar(low_stress[['Total Sleep Hours', 'Screen Time Before Bed (mins)', 'Work Hours (hrs/day)']].mean().reset_index(),
                        x='index', y=0, title="Low Stress Group")
fig_low_stress.show()

fig_optimal_sleep = px.bar(optimal_sleep[['Productivity Score', 'Mood Score', 'Stress Level']].mean().reset_index(),
                           x='index', y=0, title="Optimal Sleep Group")
fig_optimal_sleep.show()

# Print summary statistics
print("High Productivity Group:")
print(high_productivity[['Sleep Quality', 'Exercise (mins/day)', 'Caffeine Intake (mg)']].mean())

print("\nLow Stress Group:")
print(low_stress[['Total Sleep Hours', 'Screen Time Before Bed (mins)', 'Work Hours (hrs/day)']].mean())

print("\nOptimal Sleep Group:")
print(optimal_sleep[['Productivity Score', 'Mood Score', 'Stress Level']].mean())

# Sentiment analysis summary
print("\nSentiment Analysis Summary:")
print(df['Sentiment'].describe())

# Create a histogram of sentiment scores
fig_sentiment = px.histogram(df, x="Sentiment", nbins=20, title="Distribution of Sentiment Scores")
fig_sentiment.show()

# Create a scatter plot of Sentiment vs Productivity Score
fig_sentiment_prod = px.scatter(df, x="Sentiment", y="Productivity Score", color="Sleep Quality",
                                hover_data=["Total Sleep Hours", "Exercise (mins/day)"])
fig_sentiment_prod.update_layout(title="Sentiment vs Productivity Score")
fig_sentiment_prod.show()
high_productivity = df[df['Productivity Score'] > 8]
low_stress = df[df['Stress Level'] < 4]
optimal_sleep = df[(df['Total Sleep Hours'] >= 7) & (df['Total Sleep Hours'] <= 9)]
df.to_csv('processed_data.csv', index=False)