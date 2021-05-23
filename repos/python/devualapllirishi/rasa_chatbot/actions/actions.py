from typing import Text, Dict, List, Any
from rasa_sdk import Action, Tracker
from rasa_sdk.events import EventType
from rasa_sdk.executor import CollectingDispatcher

class AskForSlotAction(Action):
    def name(self) -> Text:
        return "action_ask_details"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        name = tracker.get_slot("name")
        age = tracker.get_slot("age")
        gender = tracker.get_slot("gender")
        location= tracker.get_slot("location")
        contact=tracker.get_slot("contact")
        dispatcher.utter_message(text=f"So {name}, what is your age?")
        dispatcher.utter_message(text=f"So {name},{age} what is your gender?")
        dispatcher.utter_message(text=f"So {name},{age},{gender} what is your location?")
        dispatcher.utter_message(text=f"So {name},{age},{gender},{location} what is your contact?")
        return []

class ActionSubmit(Action):
    def name(self) -> Text:
        return "action_submit"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Thank you,for your details")
        return []
