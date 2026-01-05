from flask import Flask , render_template , request , redirect
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


if __name__ == '__main__':
    app.run(debug = True)   