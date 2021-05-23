import streamlit as st
import pya3rt

apikey="DZZ9QXCfIBprCTROUjdVZQARc3AudC37"
client=pya3rt.TalkClient(apikey)

st.title("Chatbot with streamlit")#title
st.subheader("メッセージ入力してから送信をタップしてください")#header
message=st.text_input("メッセージ")#input

chat_logs=[]

def send_pya3rt():
    ans_json=client.talk(message)#input送信
    ans=ans_json['results'][0]['reply']
    chat_logs.append('you: '+message)#you
    chat_logs.append('AI: '+ans)#AI
    for chat_log in chat_logs:
        st.write(chat_log)#write

if st.button("送信"):#ボタンを押されたら以下を実行
    send_pya3rt()

