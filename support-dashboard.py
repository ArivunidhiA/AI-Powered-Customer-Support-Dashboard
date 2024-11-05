import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sqlite3
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SupportDashboard:
    def __init__(self, db_path='support_tickets.db'):
        """Initialize the Support Dashboard with database connection"""
        self.db_path = db_path
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = DecisionTreeClassifier(random_state=42)
        
    def load_data(self):
        """Load customer support ticket data from GitHub archive dataset"""
        # Using the GitHub Issues dataset as a proxy for support tickets
        url = "https://raw.githubusercontent.com/firmai/industry-machine-learning/master/support/github_issues_sample.csv"
        df = pd.read_csv(url)
        
        # Clean and prepare the data
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['closed_at'] = pd.to_datetime(df['closed_at'])
        df['resolution_time'] = (df['closed_at'] - df['created_at']).dt.total_seconds() / 3600
        
        # Store in SQLite database
        conn = sqlite3.connect(self.db_path)
        df.to_sql('tickets', conn, if_exists='replace', index=False)
        conn.close()
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for the ML model"""
        # Convert text to features
        X = self.vectorizer.fit_transform(df['title'] + ' ' + df['body'].fillna(''))
        y = df['state']  # Using state (open/closed) as our target
        return X, y
    
    def train_model(self):
        """Train the ticket classification model"""
        df = self.load_data()
        X, y = self.prepare_features(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Save the model and vectorizer
        joblib.dump(self.model, 'support_model.pkl')
        joblib.dump(self.vectorizer, 'vectorizer.pkl')
        
        # Calculate and return metrics
        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred)
    
    def get_dashboard_metrics(self):
        """Calculate key metrics for the dashboard"""
        conn = sqlite3.connect(self.db_path)
        metrics = {}
        
        # Average resolution time
        query = """
        SELECT avg(resolution_time) as avg_resolution_time,
               count(*) as total_tickets,
               sum(CASE WHEN state = 'closed' THEN 1 ELSE 0 END) * 100.0 / count(*) as resolution_rate
        FROM tickets
        """
        df = pd.read_sql_query(query, conn)
        metrics.update(df.iloc[0].to_dict())
        
        # Common issues (based on labels)
        query = """
        SELECT labels, count(*) as frequency
        FROM tickets
        GROUP BY labels
        ORDER BY frequency DESC
        LIMIT 5
        """
        metrics['common_issues'] = pd.read_sql_query(query, conn).to_dict('records')
        
        conn.close()
        return metrics
    
    def predict_resolution_time(self, title, body):
        """Predict resolution time for a new ticket"""
        text = f"{title} {body}"
        features = self.vectorizer.transform([text])
        prediction = self.model.predict(features)[0]
        return prediction

if __name__ == "__main__":
    # Initialize and train the dashboard
    dashboard = SupportDashboard()
    print("Training model...")
    model_metrics = dashboard.train_model()
    print("\nModel Performance Metrics:")
    print(model_metrics)
    
    print("\nDashboard Metrics:")
    dashboard_metrics = dashboard.get_dashboard_metrics()
    for metric, value in dashboard_metrics.items():
        if metric != 'common_issues':
            print(f"{metric}: {value:.2f}")
    
    print("\nTop 5 Common Issues:")
    for issue in dashboard_metrics['common_issues']:
        print(f"Label: {issue['labels']}, Frequency: {issue['frequency']}")
