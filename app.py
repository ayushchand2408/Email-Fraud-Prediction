import pandas as pd
import io
from flask import Flask , render_template , request , redirect , send_file , Response
import joblib
app = Flask(__name__)

# Load your Pipeline (which includes TF-IDF + Logistic Regression)
with open('Model/Email_spam_detector.pkl', 'rb') as f:
    model_pipeline = joblib.load(f)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/single_email' , methods = ['GET' , 'POST'])
def single_mail():
    prediciton_text = None

    if request.method =='POST':
        email_text = request.form.get('email_content')
        if email_text:
            prediciton = model_pipeline.predict([email_text])[0]

            prediciton_text = "Spam" if prediciton == 1 else "Not Spam"

    return render_template('one_email.html' , result = prediciton_text)


@app.route('/BulkMails', methods=['GET', 'POST'])
def bulk_mail():
    if request.method == 'POST':
        file = request.files.get('file_upload')
        if not file:
            return "No file selected"

        try:
            # 1. Read the uploaded file (works for both CSV and Excel)
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file, engine='openpyxl')
            
            # 2. Match your column heading
            target_col = 'text' 
            
            if target_col in df.columns:
                # 3. Predict
                df['Prediction'] = model_pipeline.predict(df[target_col].astype(str))
                df['Label'] = df['Prediction'].apply(lambda x: 'Spam' if x == 1 else 'Ham')

                # 4. SAVE AS CSV (This avoids the Excel 'Worksheet' error entirely)
                output = io.StringIO()
                df.to_csv(output, index=False)
                
                return Response(
                    output.getvalue(),
                    mimetype="text/csv",
                    headers={"Content-disposition": "attachment; filename=results.csv"}
                )
            else:
                return f"Column '{target_col}' not found. Please check your CSV header."

        except Exception as e:
            return f"An error occurred: {str(e)}"
                
    return render_template('bulk_email.html')

if __name__ == '__main__':
    app.run(debug = True)   