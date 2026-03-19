import os


def os_send_email(email_address, email_title, email_body):
    """
    Send an email using the system's mail command.
    > echo '{email_body}' | mail -s {email_title} {email_address}

    Parameters
    ----------
    email_address: str
        The email address to send the email to.
    email_title: str
        The title of the email.
    email_body: str
        The body content of the email.
    """
    send_email_command = f"echo '{email_body}' | mail -s {email_title} {email_address}"
    os.system(send_email_command)
