#!/bin/bash
MSMTP_CONFIG_FILE="/home/bruno/.msmtprc"




recipients="email_here"
subject="Training complete!"

echo -e "Subject:$subject\nFrom:email_here\n\nHi! training was completed. Come and join me!" | sudo -u bruno msmtp --file=$MSMTP_CONFIG_FILE -a gmail $recipients

exit 0

