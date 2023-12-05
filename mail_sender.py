import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

EMAIL_ADDRESS = os.environ['MC_EMAIL_ADDRESS']
EMAIL_PASSWORD = os.environ['MC_GMAIL_ACCESS_TOKEN']

MODULE_NAME = "Parser"

message = MIMEMultipart()
message['From'] = 'Moje články Error Trap'
message['To'] = EMAIL_ADDRESS
message['Subject'] = "Error report from Moje články %s" % MODULE_NAME

context = ssl.create_default_context()


def send_error_email(error_text):
    body = """There was an error in Python script. Please respond immidately.\n Full Error: %s""" % error_text
    message.attach(MIMEText(body, 'plain'))
    text = message.as_string()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp_server:
        try:
            smtp_server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp_server.sendmail(EMAIL_ADDRESS, EMAIL_ADDRESS, text)
        except Exception as e:
            # CANNOT SEND E-MAIL!
            raise e
