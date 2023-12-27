#!/bin/bash
MSMTP_CONFIG_FILE="/home/bruno/.msmtprc"




recipients="devil.jean42@gmail.com"
subject="Training complete!"

echo -e "Subject:$subject\nFrom:devil.jean42@gmail.com\n\nHi! training was completed. Come and join me!" | sudo -u bruno msmtp --file=$MSMTP_CONFIG_FILE -a gmail $recipients

exit 0

