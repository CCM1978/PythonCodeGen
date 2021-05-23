#https://github.com/settings/tokens
#just for "public repo"
#token: ghp...etc
#pip3 install pygithub

from github import Github
import time
from datetime import datetime
import os

end_time = time.time()
start_time = end_time-86400

#ACCESS_TOKEN = open("token.txt","r").read()
g = Github('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')
print(g.get_user())

for i in range(3):
    try:
        start_time_str = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d')
        end_time_str = datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d')
        query = f"chatbot language:javascript created:{start_time_str}..{end_time_str}"
        end_time -= 86400
        start_time -= 86400
        result = g.search_repositories(query)
        print(result.totalCount)

        for repository in result:
            print(f"{repository.clone_url}")
            #print(dir(repository))

            os.system(f"git clone {repository.clone_url} repos/javascript/{repository.owner.login}/{repository.name}")
            print(f"Current start time {start_time}")

    except Exception as e:
        print(str(e))
        print("Script encountered an error...")
        time.sleep(120)

print("Finished. Your new end time should be:", start_time)