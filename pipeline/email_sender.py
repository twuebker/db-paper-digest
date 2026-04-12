"""Send the digest as an HTML email via Gmail SMTP."""

import os
import re
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def send_digest_email(config: dict, html_body: str, digest_date: date, n_papers: int) -> None:
    subject = f"DB Paper Digest \u2013 {digest_date} | {n_papers} new paper{'s' if n_papers != 1 else ''}"
    _send(config, subject, html_body)


def send_empty_email(config: dict, digest_date: date) -> None:
    subject = f"DB Paper Digest \u2013 {digest_date} | nothing new"
    html = f"<p>No new database papers found for {digest_date}.</p>"
    _send(config, subject, html)


def _send(config: dict, subject: str, html: str) -> None:
    sender = config["sender_email"]
    recipient = config["recipient_email"]
    password = os.environ["GMAIL_APP_PASSWORD"]

    plain = _html_to_plain(html)
    msg = _build_message(sender, recipient, subject, html, plain)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(sender, password)
        smtp.sendmail(sender, recipient, msg.as_string())

    print(f"[email] Sent: {subject!r} → {recipient}")


def _build_message(sender: str, recipient: str, subject: str, html: str, plain: str) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(plain, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))
    return msg


def _html_to_plain(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).strip()
