# Download the helper library from https://www.twilio.com/docs/python/install
from twilio.rest import Client


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
##account_sid = os.environ['AC819370f38cd2520cf11d38e6fe710f03']
##auth_token = os.environ['44601c21256cb6bf415d6deb762edc46']
account_sid = 'AC819370f38cd2520cf11d38e6fe710f03'
auth_token = '44601c21256cb6bf415d6deb762edc46'
client = Client(account_sid, auth_token)

message = client.messages \
                .create(
                     body="irregular walking patterns were detected",
                     from_='+18559603458',
                     to='+15714356878'
                 )

##print(message.sid)
                