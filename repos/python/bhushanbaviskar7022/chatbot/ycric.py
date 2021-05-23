import cricapi
import json
apikey='ce5hc9ToGjWV4qRA4fmg9l5NJTv1'
criapi = cricapi.Cricapi(apikey)

def test():
  params = {'unique_id':1198242}

  liveM=criapi.fantasySummary(params)
  print(json.dumps(liveM['data']['team'],indent=4))
  #print(json.dumps(liveM,indent=4))
#test()
def score(matchNumber):
  params = {'unique_id':matchNumber}
  team=criapi.fantasySummary(params)
  t=[]
  player=[]
  for i in team['data']['batting']:
    #print(i['title'])
    player.append(i['title']+':\n ')
    for j in i['scores']:
      t=j['batsman']
      if j["dismissal-info"]=="":
        t=t+"* "+str(j['R'])+'('+str(j['B'])+') SR:'+str(j['SR'])+'\n'
      else:
        t=t+" "+j["dismissal-info"]+" "+str(j['R'])+'('+str(j['B'])+') SR:'+str(j['SR'])+'\n'


      player.append(t)
    player.append('\n \n')
  player="\n".join(player)
  #print(player)
  return {"fulfillmentMessages": [
      {
        "quickReplies": {
          "title": player,
          "quickReplies": ['Score Board','Squad','Upcoming Matches']
        },
        "platform": "FACEBOOK"
      },
      {
        "text": {
          "text": [
            player
          ]
        }
      }]}

#print(score(1198242))
def Teams(matchNumber):
  params = {'unique_id':matchNumber}
  team=criapi.fantasySummary(params)
  t=[]
  player=[]
  for i in team['data']['team']:
    t.append(i['name'])
    player.append(i['name']+': ')
    for j in i['players']:
      player.append(j['name'])
    player.append('\n')
  player="\n".join(player)
  #print(player)
  return {"fulfillmentMessages": [
      {
        "quickReplies": {
          "title": player,
          "quickReplies": ['Score Board','Live Matches','Upcoming Matches']
        },
        "platform": "FACEBOOK"
      },
      {
        "text": {
          "text": [
            player
          ]
        }
      }]}
#print(Teams(1198243))





def upComing():
  params = {'unique_id':1225249}

  upCom=criapi.matches()
  matches=[]
  #print(json.dumps(upCom,indent=4))
  for i in upCom['matches']:
      matches.append(i['team-1']+' '+'vs'+' '+i['team-2']+' ('+i['type']+')'+'\n'+'Start on: '+(i['date'][:10].replace('-','/')))
  matches="\n\n".join(matches)
  return {"fulfillmentMessages": [
      {
        "quickReplies": { 
          "title": matches,
          "quickReplies": ['Live Matches']
        },
        "platform": "FACEBOOK"
      },
      {
        "text": {
          "text": [
            matches
          ]
        }
      }]}
#print(upComing())


def liveScore(matchNumber):
    params = {'unique_id':matchNumber}
    matchScore=criapi.cricketScore(params)
    #print(matchNumber)
    #print(json.dumps(matchScore,indent=4))
    #print(matchScore["matchStarted"])
    if not matchScore["matchStarted"]:
      return   {
      "fulfillmentMessages": [
      {
        "payload": {
          "facebook": {
            "attachment": {
              "type": "template",
              "payload": {
                "buttons": [{"type": "postback",
                  "payload":str(matchNumber),
                  "title":"Refresh"
                  }],
                "text": 'Match Not Started',
                "template_type": "button"
              }
            }
          }
        },
        "platform": "FACEBOOK"
      },
      {
        "text": {
          "text": [
            'Match Not Started'
          ]
        }
      }
    ]
  }



    #title='Match ID: '+str(matchNumber)+'\n'
    title=matchScore["team-1"]+' vs '+matchScore["team-2"]+'\n'+matchScore["stat"]+'\n\n'+str(matchScore['score']).replace('&amp;','&')
    return {"fulfillmentMessages": [
      {
        "quickReplies": {
          "title": title,
          "quickReplies": ['Score Board','Squads']
        },
        "platform": "FACEBOOK"
      },
      {
        "text": {
          "text": [
            title
          ]
        }
      }]}
    '''matchListD.append({"type": "postback",
          "payload":i['unique_id'],
          "title": temp['team-1']+' vs '+temp['team-2']
          })
        matchListS.append(temp['team-1']+' vs '+temp['team-2']+" "+i['unique_id'])
    matchListS="\n".join(matchListS)'''
#liveScore('1225249')
def liveMatches():
    liveM=criapi.cricket()
    matchListS=[]
    matchListD=[]
    title="Select Live Matches"
    for i in liveM['data']:
        params = {'unique_id':i['unique_id']}
        #print("dgxd",params)
        temp=criapi.cricketScore(params)
        if not temp["matchStarted"]:
          continue
        #print(temp)
        matchListD.append({"type": "postback",
          "payload":i['unique_id'],
          "title": temp['team-1']+' vs '+temp['team-2']
          })
        matchListS.append(" "+i['unique_id']+" "+temp['team-1']+' vs '+temp['team-2'])
    matchListS="\n".join(matchListS)
    if not len(matchListD):
      matchListD=[{"type": "postback",
            "payload":"Live Matches",
            "title":"Refresh"
            }]
      matchListS="Sorry.. No Match in progress.."
      title=matchListS
    return {
      "fulfillmentMessages": [
      {
        "payload": {
          "facebook": {
            "attachment": {
              "type": "template",
              "payload": {
                "buttons": matchListD+[{"type": "postback",
            "payload":"upcoming match",
            "title":"Upcoming Matches"
            }],
                "text": title,
                "template_type": "button"
              }
            }
          }
        },
        
        "platform": "FACEBOOK"
      },
      {
        "text": {
          "text": [
            matchListS
          ]
        }
      }
    ]
  }

