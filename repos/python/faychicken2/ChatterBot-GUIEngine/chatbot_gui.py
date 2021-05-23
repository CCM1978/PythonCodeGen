import wx
from wx.adv import Animation, AnimationCtrl
from typing import Callable


# main wx frame object
class Main(wx.Frame):
    def __init__(self, parent, title: str, gui: 'ChatbotGUI'):
        # init parent
        wx.Frame.__init__(self, parent, -1, title=title)

        # a reference to the chatbot GUI
        self.gui = gui

        # grid for splitting the screen into two parts, the gif and I/O elements
        self.grid = wx.BoxSizer(wx.HORIZONTAL)

        # user & AI message history
        self.user_message_history = []
        self.ai_message_history = []

        # panel for all of the I/O elements
        self.io_panel = wx.Panel(self)

        # the sizer for the panel I/O elements
        self.io_sizer = wx.BoxSizer(wx.VERTICAL)

        # chat bot animation asset
        self.chatbot_gif = Animation('ChatbotFemale.gif')

        # animation controller for the chat bot gif
        self.chatbot_control = AnimationCtrl(self, -1, self.chatbot_gif)

        #
        # I/O elements
        #

        # input chat
        self.input_chat = wx.TextCtrl(self.io_panel, size=wx.Size(200, 200), style=wx.TE_READONLY | wx.TE_MULTILINE)

        # input chat label
        self.input_chat_label = wx.StaticText(self.io_panel, label="Chat with AI:")

        # input chat text box
        self.input_box = wx.TextCtrl(self.io_panel, style=wx.TE_PROCESS_ENTER, size=wx.Size(200, 20))

        # input chat enter button
        self.chat_button = wx.Button(self.io_panel, label="Send")

        # input chat label
        self.output_chat_label = wx.StaticText(self.io_panel, label="AI Responses:")

        # output chat
        self.output_chat = wx.TextCtrl(self.io_panel, size=wx.Size(200, 200), style=wx.TE_READONLY | wx.TE_MULTILINE)

        # start AI gif button
        self.start_gif = wx.Button(self.io_panel, label="Start GIF", size=wx.Size(200, 100))

        # stop gif button
        self.stop_gif = wx.Button(self.io_panel, label="Stop GIF", size=wx.Size(200, 100))

        #
        #   Add and size elements
        #

        # add elements to the I/O sizer
        self.io_sizer.Add(self.input_chat, 0, wx.EXPAND | wx.ALL, 5)
        self.io_sizer.Add(self.input_chat_label, 0, wx.EXPAND | wx.LEFT | wx.TOP, 5)
        self.io_sizer.Add(self.input_box, 0, wx.EXPAND | wx.ALL, 5)
        self.io_sizer.Add(self.chat_button, 0, wx.EXPAND | wx.ALL, 5)
        self.io_sizer.Add(self.output_chat_label, 0, wx.EXPAND | wx.LEFT | wx.TOP, 5)
        self.io_sizer.Add(self.output_chat, 0, wx.EXPAND | wx.ALL, 5)
        self.io_sizer.Add(self.start_gif, 0, wx.EXPAND | wx.ALL, 5)
        self.io_sizer.Add(self.stop_gif, 0, wx.EXPAND | wx.ALL, 5)

        # add elements to the main grid sizer
        self.grid.Add(self.io_panel, 0, wx.EXPAND | wx.ALL)
        self.grid.Add(self.chatbot_control)

        # size and fit the sizers
        self.io_panel.SetSizerAndFit(self.io_sizer)
        self.SetSizerAndFit(self.grid)

        #
        #   Bind buttons to functions
        #
        self.Bind(wx.EVT_TEXT_ENTER, self.on_send_press)
        self.Bind(wx.EVT_BUTTON, self.on_send_press, self.chat_button)
        self.Bind(wx.EVT_BUTTON, self.start_animation, self.start_gif)
        self.Bind(wx.EVT_BUTTON, self.stop_animation, self.stop_gif)

    def start_animation(self, event):
        print(event)
        self.chatbot_control.Play()

    def stop_animation(self, event):
        self.chatbot_control.Stop()

    # updates the user and AI message histories
    def update_message_history(self):
        # variables to store the aggregated text
        user_text = ""
        ai_text = ""

        # aggregate user messages
        for message in self.user_message_history:
            user_text += "You> " + message + "\n"

        # aggregate ai messages
        for message in self.ai_message_history:
            ai_text += "AI> " + str(message) + "\n"

        # update the chats
        self.input_chat.SetValue(user_text)
        self.output_chat.SetValue(ai_text)

    # send a ai message
    def send_ai_message(self, text: str):
        # add the message to message history
        self.ai_message_history.insert(0, text)

        # update the message history
        self.update_message_history()

    # clears the user and AI chat history
    def clear_chat(self):
        self.user_message_history = []
        self.ai_message_history = []
        self.update_message_history()

    # function handling "send" button press
    def on_send_press(self, event):
        # read the text box
        text = self.input_box.GetValue()
        if text == "":
            return

        # clear the text box
        self.input_box.SetValue("")

        # add the message to message history
        self.user_message_history.insert(0, text)

        # update the message history
        self.update_message_history()

        # call the message handler function for the ChatBot GUI
        self.gui.call_on_message(text)


# main program class, controls the GUI and interactions with the GUI
class ChatbotGUI:
    def __init__(self, title: str):
        # app object
        self.app = wx.App()

        # main frame
        self.frame = Main(None, title, self)

    # clear's chat history
    def clear(self):
        self.frame.clear_chat()

    # starts gif
    def start_gif(self):
        self.frame.start_animation(None)

    # starts gif
    def stop_gif(self):
        self.frame.stop_animation(None)

    # sends a message as the AI to the chat
    def send_ai_message(self, text: str):
        self.frame.send_ai_message(text)

    # handles passing incoming user messages to the on_message handler
    def call_on_message(self, text: str):
        if getattr(self, "on_message", None) is None:
            print("Please define the 'on_message' function!")
            return

        # call the on_message handler
        getattr(self, "on_message")(self, text)

    # used to easily define the on_message handler function
    def event(self, coroutine):
        # handle general on_connect, and on_disconnect handlers
        if coroutine.__name__ == "on_message":
            # replaces the existing coroutine with the provided one
            setattr(self, coroutine.__name__, coroutine)
            return True
        return False

    # run the chat bot GUI
    def run(self) -> None:
        self.frame.Show()
        self.app.MainLoop()
