# coding: UTF-8
import unirest

# These code snippets use an open-source library.
# These code snippets use an open-source library.
response = unirest.get("https://community-angellist.p.mashape.com/follows/batch?ids=86500%2C173917",
  headers={
    "X-Mashape-Key": "cmfsqCqjjvmsh5ngHwiOaTKih9SYp153DQjjsnmXGIPnRHSQGM",
    "Accept": "application/json"
  }
)
print response