# # This files contains your custom actions which can be used to run
# # custom Python code.
# #
# # See this guide on how to implement these action:
# # https://rasa.com/docs/rasa/custom-actions


# # This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List, Union

# from rasa_sdk import Action, Tracker
# from rasa_sdk import Tracker, FormValidationAction
# from rasa_sdk.executor import CollectingDispatcher
# from rasa_sdk.events import SlotSet, UserUtteranceReverted
# from rasa_sdk.forms import FormAction
# from rasa_sdk.types import DomainDict
# from datetime import datetime
# import pytz
# from rasa_sdk.forms import FormValidationAction

# # ================================================================================================================================


# class BookTableInfo(FormAction):
#     def name(self) -> Text:
#         return "form_book_table"

#     @staticmethod
#     def required_slots(tracker: Tracker) -> List[Text]:
#         return ["number", "name", "time"]

#     def submit(
#             self,
#             dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any],
#     ) -> List[Dict]:

#         # utter submit template
#         dispatcher.utter_message(template="utter_submit", number=tracker.get_slot('number'),
#                                  name=tracker.get_slot('name'),
#                                  time=tracker.get_slot('time'),)
#         return []

#     def slot_mappings(self) -> Dict[Text, Union[Dict, List[Dict]]]:

#         return {
#             "number": self.from_entity(entity="number", intent='number'),
#             "name": self.from_entity(entity="name", intent="table_type"),
#             "time": self.from_entity(entity="time", intent="time")
#         }

# class ResetSlots(Action):

#     def name(self):
#         return "action_reset_slots"

#     def run(self, dispatcher, tracker, domain):
#         return [SlotSet("number", None), SlotSet("name", None), SlotSet("time", None)]





# class ValidateRestaurantForm(FormValidationAction):

#     def name(self) -> Text:
#         return "validate_form_book_table"


#     def validate_time(
#         self,
#         slot_value: Any,
#         dispatcher: CollectingDispatcher,
#         tracker: Tracker,
#         domain: DomainDict,
#     ) -> Dict[Text, Any]:

#         user_time = tracker.get_slot("time")

#         IST = pytz.timezone('Asia/Kolkata')
#         datetime_ist = datetime.now(IST)
#         sys_time = datetime_ist.strftime("%H:%M")
#         print(sys_time)
#         sys_arr = []
#         sys_arr = sys_time.split(':')
# #user_arr = user_time.split(':')
#         for i in range(0, len(sys_arr)):
#             sys_arr[i] = int(sys_arr[i])
#             #user_arr[i] = int(user_arr[i])
#         print(sys_arr)
#         # print(user_arr)
#         check = 0
#         flag = 0
#         if(sys_arr[1] < 30 and sys_arr[1] > 0):
#             sys_arr[1] = 30
#         if(sys_arr[1] > 30 and sys_arr[1] < 59):
#             sys_arr[1] = 0
#             sys_arr[0] = sys_arr[0] + 1
#         print(sys_arr)
#         if(user_time.__contains__('In an hour')):
#             flag = 1
#             sys_arr[0] = sys_arr[0] + 1
#             check_Var = sys_arr[0]
#             if((check_Var >= 7 and check_Var < 10) or (check_Var >= 19 and check_Var < 22)):
#                 print("Checking hour ")
#                 check = 1
#             else:
#                 check = 0
#             if(check == 1):
#                 print("Your order can be booked on hour = ", str(sys_arr[0]), "and minute = ", str(sys_arr[1]))
#                 message = 'You have reserved {number} seats in our {name} section for ' + str(sys_arr[0]) + str(sys_arr[1]) +' Thanks!'
#             else:
#                 print("Your order can not be booked on hour =", str(sys_arr[0]), "and minute = ", str(sys_arr[1]))
#                 message = 'We are not open at that time. We are only open from 7pm to 10pm'

#         if(user_time.__contains__('In half an hour') and flag == 0):
#             flag = 1
#             check = 0
#             sys_arr[1] = sys_arr[1] + 30
#             if(sys_arr[1] > 59):
#                 sys_arr[0] = sys_arr[0] + 1
#                 sys_arr[1] = sys_arr[1] - 60
#                 check_Var = sys_arr[0]
#         if((check_Var >= 7 and check_Var < 10) or (check_Var >= 19 and check_Var < 22)):
#             print("Checking half an hour")
#             check = 1
#         else:
#             print("checking not")
#             check = 0
#         if(check == 1):
#             print("Your order can be booked on hour = ", str(sys_arr[0]), "and minute = ", str(sys_arr[1]))
#             message = 'You have reserved {number} seats in our {name} section for ' + str(sys_arr[0]) + str(sys_arr[1]) +' Thanks!'
#         else:
#             print("Your order can not be booked on hour =", str(sys_arr[0]), "and minute = ", str(sys_arr[1]))
#             message = 'We are not open at that time. We are only open from 7pm to 10pm'

#         if(flag == 0):
#             # that means user had entered the time in HH:MM format only
#             # now we will add minutes in the time given by user
#             user_arr = user_time.split(':')
#             for i in range(0, len(sys_arr)):
#                 user_arr[i] = int(user_arr[i])
#             if(user_arr[1] < 30 and user_arr[1] > 0):
#                 user_arr[1] = 30
#             if(user_arr[1] > 30 and user_arr[1] < 59):
#                 user_arr[1] = 0
#                 user_arr[0] = user_arr[0] + 1

#             check_Var = user_arr[0]
#             if((check_Var >= 7 and check_Var < 10) or (check_Var >= 19 and check_Var < 22)):
#                 check = 1
#             else:
#                 check = 0
#             if(check == 1):
#                 print("Your order can be booked on hour = ", str(user_arr[0]), "and minute = ", str(user_arr[1]))
#                 message = 'You have reserved {number} seats in our {name} section for ' + str(sys_arr[0]) + str(sys_arr[1]) +' Thanks!'
#             else:
#                 print("Your order can not be booked on hour =", str(user_arr[0]), "and minute = ", str(user_arr[1]))
#                 message = 'We are not open at that time. We are only open from 7pm to 10pm'



#         dispatcher.utter_message(message, number=tracker.get_slot("number"),
#                          name=tracker.get_slot("name"), 
#                          time=str(user_arr[0])+':'+ str(user_arr[1]))

#         slot_set("time", str(user_arr[0])+':'+ str(user_arr[1]))
#         return{"time":str(user_arr[0])+':'+ str(user_arr[1])}

                       

# # =============================================================================================================================


# # class CheckWorkingHours(Action):
# #     def name(self): # type: () -> Text
# #         return "time_Managing"

# #     def run(self, dispatcher: CollectingDispatcher,
# #             tracker: Tracker,
# #             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

# #         IST=pytz.timezone('Asia/Kolkata')
# #         datetime_ist=datetime.now(IST)
# #         now = datetime_ist.strftime('%H:%M')
# #         # today19pm = now.replace(hour=19, minute=0, second=0, microsecond=0)
# #         # today22pm = now.replace(hour=22, minute=0, second=0, microsecond=0)
# #         time =tracker.get_slot("time")

# # #_____________For User Input____________________

# #         new_str1 = time[:-2]
# #         am_pm_Var = time[-2:]
# #         arr1 = []
# #         arr1 = new_str1.split(':')
# #         print(arr1)
# #         print(new_str1) #for am and pm
# #         print(am_pm_Var)
# #         for i in range(0,len(arr1)):
# #             arr1[i]=int(arr1[i])

# #         print(arr1)

# # #_____________For Current Input____________________

# #       #  new_str2 = now[:-2]   #____for String ___
# #        # am_pm_Var = now[-2:]  #____for am and pm ___
# #         arr2 = []
# #         print(arr2)
# #       #  print(new_str2)
# #       #  print(am_pm_Var)
# #         for i in range(0,len(arr2)):
# #             arr2[i] = int(arr2[i])


# #         print(arr2)

# # #_____________For Current Input____________________


# #         if(time=='hour'):
# #             x=arr2[0]+1
# #             if(x>=7 or x<10):
# #                 if(arr2[1]>1 and arr2[1]<30):
# #                     z=str(arr2[0])+ ': 30 pm'
# #                     message='You have reserved {number} seats in our {name} section for {z} Thanks!'
# #                 else:
# #                     x=arr2[0]+1
# #                     z=str(arr2[0])+ ': 00 pm'
# #                     message='You have reserved {number} seats in our {name} section for {z} Thanks!'
# #             else:
# #                 message = 'We are not open at that time. We are only open from 7pm to 10pm'

# #         if(time=='half'):
# #             x1=arr2[1]+30
# #             if(arr2[1]>60):
# #                 x1=arr2[1]-60
# #                 y1=arr2[0]+1

# #                 if(y1>=7 or y1<10):
# #                     if(arr2[1]>1 and arr2[1]<30):
# #                         z=str(arr2[0])+ ': 30 pm'
# #                         message='You have reserved {number} seats in our {name} section for {z} Thanks!'
# #                     else:
# #                         x=arr2[0]+1
# #                         z=str(arr2[0])+ ': 00 pm'
# #                         message='You have reserved {number} seats in our {name} section for {z} Thanks!'

# #                 else:
# #                     message = 'We are not open at that time. We are only open from 7pm to 10pm'


# #         if(7<=arr1[0]<10):
# #             if(arr1[1]>1 and arr1[1]<30):
# #                 z=str(arr1[1])+ ': 30 pm'
# #                 message='You have reserved {number} seats in our {name} section for {z} Thanks!'
# #             else:
# #                 x=arr1[0]+1
# #                 z=str(arr1[0])+ ': 00 pm'
# #                 message='You have reserved {number} seats in our {name} section for {z} Thanks!'
# #         else:
# #             message = 'We are not open at that time. We are only open from 7pm to 10pm'


# #         # print(time)
# #         # if today19pm <= time < today22pm:
# #         #     message = 'You have reserved {number} seats in our {name} section for {time} Thanks!'
# #         # else:
# #         #     message = 'We are not open at that time. We are only open from 7pm to 10pm'

# #         dispatcher.utter_message(message)

# #         return []

# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions
# This is a simple example for a custom action which utters "Hello World!"
from typing import Any, Text, Dict, List, Union
from rasa_sdk import Action, Tracker
from rasa_sdk import Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, UserUtteranceReverted
from rasa_sdk.forms import FormAction
from rasa_sdk.types import DomainDict
from datetime import datetime
import pytz
from rasa_sdk.forms import FormValidationAction
# ================================================================================================================================
# class BookTableInfo(FormAction):
#     def name(self) -> Text:
#         return "form_book_table"
#     def submit(
#             self,
#             dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any],
#     ) -> List[Dict]:
#         # utter submit template
#         dispatcher.utter_message(template="utter_submit", number=tracker.get_slot('number'),
#                                  name=tracker.get_slot('name'),
#                                  time=tracker.get_slot('time'),)
#         return []
class ResetSlots(Action):
    def name(self):
        return "action_reset_slots"
    def run(self, dispatcher, tracker, domain):
        return [SlotSet("number", None), SlotSet("name", None), SlotSet("time", None)]
# =============================================================================================================================

class ValidateRestaurantForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_form_book_table"

    def validate_time(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        xestTime=""
        count=0
        user_time = tracker.get_slot("time")
        IST = pytz.timezone('Asia/Kolkata')
        datetime_ist = datetime.now(IST)
        sys_time = datetime_ist.strftime("%H:%M")
        print(sys_time)
        sys_arr = []
        sys_arr = sys_time.split(':')
        for i in range(0, len(sys_arr)):
            sys_arr[i] = int(sys_arr[i])
        print(sys_arr)
        check = 0
        flag = 0
        if(sys_arr[1] < 30 and sys_arr[1] > 0):
            sys_arr[1] = 30
        if(sys_arr[1] > 30 and sys_arr[1] < 59):
            sys_arr[1] = 0
            sys_arr[0] = sys_arr[0] + 1
        print(sys_arr)
        if(user_time.__contains__('In an hour')):
            flag = 1
            sys_arr[0] = sys_arr[0] + 1
            check_Var = sys_arr[0]
            if((check_Var >= 7 and check_Var < 10) or (check_Var >= 19 and check_Var < 22)):
                print("Checking hour ")
                check = 1
            else:
                check = 0
            if(check == 1):
                print("Your order can be booked on hour = ", str(sys_arr[0]), "and minute = ", str(sys_arr[1]))
                count=count+1
                xestTime=str(sys_arr[0])+":"+str(sys_arr[1])
                #message = 'You have reserved {number} seats in our {name} section for ' + str(sys_arr[0]) +':'+ str(sys_arr[1]) +'pm Thanks!'
            else:
                print("Your order can not be booked on hour =", str(sys_arr[0]), "and minute = ", str(sys_arr[1]))
                #message = 'We are not open at that time. We are only open from 7pm to 10pm '
        if(user_time.__contains__('In half an hour') and flag == 0):
            flag = 1
            check = 0
            sys_arr[1] = sys_arr[1] + 30
            if(sys_arr[1] > 59):
                sys_arr[0] = sys_arr[0] + 1
                sys_arr[1] = sys_arr[1] - 60
            check_Var = sys_arr[0]
            if((check_Var >= 7 and check_Var < 10) or (check_Var >= 19 and check_Var < 22)):
                print("Checking half an hour")
                check = 1
            else:
                print("checking not")
                check = 0
            if(check == 1):
                print("Your order can be booked on hour = ", str(sys_arr[0]), "and minute = ", str(sys_arr[1]))
                count=count+1
                xestTime=str(sys_arr[0])+":"+str(sys_arr[1])
               # message = 'You have reserved {number} seats in our {name} section for ' + str(sys_arr[0]) +':'+ str(sys_arr[1]) +' pm Thanks!'
            else:
                print("Your order can not be booked on hour =", str(sys_arr[0]), "and minute = ", str(sys_arr[1]))
               # message = 'We are not open at that time. We are only open from 7pm to 10pm '
                
                
                
        if(flag == 0):
            # that means user had entered the time in HH:MM format only
            # now we will add minutes in the time given by user
            user_time1 = "" 
            user_time1 = user_time 
            user_time  = user_time[:-3] # 9:30 am

            user_arr = user_time.split(':')
            for i in range(0, len(sys_arr)):
                user_arr[i] = int(user_arr[i])
            if(user_arr[1] < 30 and user_arr[1] > 0):
                user_arr[1] = 30
            if(user_arr[1] > 30 and user_arr[1] < 59):
                user_arr[1] = 0
                user_arr[0] = user_arr[0] + 1
            check_Var = user_arr[0]
            if((check_Var >= 7 and check_Var < 10) or (check_Var >= 19 and check_Var < 22)):
                check = 1
            else:
                check = 0
            if(check == 1):
                print("Your order can be booked on hour = ", str(user_arr[0]), "and minute = ", str(user_arr[1]))
                count=count+1
                xestTime=str(user_arr[0])+":"+str(user_arr[1])
               # message = 'You have reserved {number} seats in our {name} section for ' + str(user_arr[0]) +':'+ str(user_arr[1]) +' pm Thanks!'
            else:
                print("Your order can not be booked on hour =", str(user_arr[0]), "and minute = ", str(user_arr[1]))
               # message = 'We are not open at that time. We are only open from 7pm to 10pm '
                

        if(count==1):
            dispatcher.utter_message(response="utter_submit", number=tracker.get_slot("number"),
                                 name=tracker.get_slot("name"), 
                                 time=xestTime)
            return{"time":user_time}
        else:
            message = 'We are not open at that time. We are only open from 7pm to 10pm '
            dispatcher.utter_message(message)
            return{"time":None}