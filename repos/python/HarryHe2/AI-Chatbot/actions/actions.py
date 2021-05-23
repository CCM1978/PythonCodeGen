# https://rasa.com/docs/rasa/custom-actions

from typing import Any, Text, Dict, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction 
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.events import SlotSet



class totalOrder(Action):

    revenue = 0

    def name(self) -> Text:
        return "action_order_confirmed"


    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain:Dict[Text, Any])-> List[Dict[Text, Any]]:

        price_list = {"big mac": 6, "cheeseburger": 5, "bacon angus": 5, "classic angus": 6, "mcspicy": 6,
              "chicken salad": 12, "garden salad": 5, "chicken nuggets": 10, "fries": 5, "apple pie": 2,
              "hash brown": 2, "strawberry sundae": 3, "coke": 3, "sprite": 3, "ice latte": 3, "apple juice": 2}

        num_list = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4,"5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
                    "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15}

        
        food = tracker.get_slot("food")
        food_num = tracker.get_slot("food_num")

        print("food = {}".format(food))
        print("food_num = {}".format(food_num))

        drinks = tracker.get_slot("drinks")
        drinks_num = tracker.get_slot("drinks_num")

        print("drinks = {}".format(drinks))
        print("drinks_num = {}".format(drinks_num))

        food_total = 0
        drinks_total = 0

        # calculate food price
        if food is None:
            food_total = 0

        else:
            if isinstance(food,list):
                for index in range(len(food)):
                    food_price = price_list[food[index].lower()]
                    total = food_price * num_list[food_num[index]]
                 
                    food_total += total
            else:
                food_price = price_list[food.lower()]
                total = food_price * num_list[food_num]

                food_total += total
       
        # calculate drinks price
        if drinks is None:
            drinks_total = 0

        else:
            if isinstance(drinks,list):
                for indexx in range(len(drinks)):
                    drinks_price = price_list[drinks[indexx].lower()]
                    total_drinks = drinks_price * num_list[drinks_num[indexx]]

                    drinks_total += total_drinks
        
            else:
                drinks_price = price_list[drinks.lower()]
                total_drinks = drinks_price * num_list[drinks_num]

                drinks_total += total_drinks

        
        order_total = food_total + drinks_total
        
        self.revenue += order_total

        print("order total: {}".format(order_total))
        print(f"revenue: {self.revenue}")


        return [SlotSet("TOTAL",order_total),SlotSet("revenue", self.revenue),
                SlotSet("food",None),SlotSet("food_num",None),
                SlotSet("drinks",None),SlotSet("drinks_num",None)]


class showOrder(Action):

    def name(self) -> Text:
        return "action_show_order"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain:Dict[Text, Any])-> List[Dict[Text, Any]]:


        # show food 
        ###################################################################

        food = tracker.get_slot("food")
        food_num = tracker.get_slot("food_num")

        print("food = {}".format(food))

        print("food_num = {}".format(food_num))

        message ="Sure! Here is your order:"

        if food is None:
            message += "\nNo food"

        else:
            if isinstance(food,list):
                for index in range(len(food)):
                    message += f"\n{food_num[index]} {food[index]}"

            else:
                message += f"\n{food_num} {food}"


        # show drinks
        ###################################################################


        drinks = tracker.get_slot("drinks")
        drinks_num = tracker.get_slot("drinks_num")

        print("drinks = {}".format(drinks))

        print("drinks_num = {}".format(drinks_num))

        if drinks is None:
            message += "\nNo drinks"

        else:
            if isinstance(drinks,list):
                for indexx in range(len(drinks)):
                    message += f"\n{drinks_num[indexx]} {drinks[indexx]}"

            else:
                message += f"\n{drinks_num} {drinks}"

        message +="\nconfirm your order by replying 'yes', otherwiese feel free to change or add anything"


        dispatcher.utter_message(text=message)

        return []


class showRevenue(Action):

    def name(self) -> Text:
        return "action_show_revenue"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain:Dict[Text, Any])-> List[Dict[Text, Any]]:

        entered_ID = tracker.get_slot("ID")
        print(entered_ID)

        revenue = tracker.get_slot("revenue")
        print(revenue)

        verified_ID = "imtheboss"

        if entered_ID == verified_ID:
            dispatcher.utter_message(text=f"The revenue for today is {revenue}")

        elif entered_ID != verified_ID:
            dispatcher.utter_message("Sorry, admin code is incorrect. Access denied.")

        return []







class addOrder(Action):

    def name(self) -> Text:
        return "action_add_order"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain:Dict[Text, Any])-> List[Dict[Text, Any]]:


        food_list = ["big mac", "cheeseburger", "bacon angus", "classic angus", "mcspicy",
              "chicken salad", "garden salad", "chicken nuggets", "fries", "apple pie", "hash brown"]

        drinks_list = ["strawberry sundae", "coke", "sprite", "ice latte", "apple juice"]

        food = tracker.get_slot("food")
        food_num = tracker.get_slot("food_num")

        drinks = tracker.get_slot("drinks")
        drinks_num = tracker.get_slot("drinks_num")

        add = tracker.get_slot("add")
        add_num = tracker.get_slot("add_num")

        add_food = [] 
        add_food_num = []

        add_drinks = []
        add_drinks_num = []

        if isinstance(add,list): # add is a list
            for num in range(len(add)):
                if add[num].lower() in food_list:
                    add_food.append(add[num].lower())
                    add_food_num.append(add_num[num])

                else:
                    add_drinks.append(add[num].lower())
                    add_drinks_num.append(add_num[num])

        else: # add is a single value
            if add.lower() in food_list:
                add_food.append(add.lower())
                add_food_num.append(add_num)

            else:
                add_drinks.append(add.lower())
                add_drinks_num.append(add_num)

        # modify food slot

        # food is None ####################################################################

        if food is None: 

            # SlotSet("food",add_food)
            # SlotSet("food_num",add_food_num)

            # modify drinks slot

            # drinks is None
            if drinks is None: 
                # SlotSet("drinks",add_drinks)
                # SlotSet("drinks_num",add_drinks_num)
                return [SlotSet("food",add_food),SlotSet("food_num",add_food_num),
                        SlotSet("drinks",add_drinks),SlotSet("drinks_num",add_drinks_num)]

            # drinks is a single value
            elif isinstance(drinks,str): 
                add_drinks.append(drinks)
                add_drinks_num.append(drinks_num)

                print(f"add_drinks: {add_drinks}")
                print(f"add_drinks_num: {add_drinks_num}")

                # SlotSet("drinks",add_drinks)
                # SlotSet("drinks_num",add_drinks_num)
                return [SlotSet("food",add_food),SlotSet("food_num",add_food_num),
                        SlotSet("drinks",add_drinks),SlotSet("drinks_num",add_drinks_num)]

            # drinks is a list
            else: 
                for indexx in range(len(add_drinks)):
                    drinks.append(add_drinks[indexx]) # modify drinks list
                    drinks_num.append(add_drinks_num[indexx]) # modify drinks_num list

                print(f"change_drinks: {drinks}")
                print(f"change_drinks_num: {drinks_num}")

                # SlotSet("drinks",drinks)
                # SlotSet("drinks_num",drinks_num)
                return [SlotSet("food",add_food),SlotSet("food_num",add_food_num),
                        SlotSet("drinks",drinks),SlotSet("drinks_num",drinks_num)]


        # food is a single value ####################################################################

        elif isinstance(food,str): 
            add_food.append(food)
            add_food_num.append(food_num)

            print(f"add_food: {add_food}")
            print(f"add_food_num: {add_food_num}")

            # SlotSet("food",add_food)
            # SlotSet("food_num",add_food_num)


            # modify drinks slot

            # drinks is None
            if drinks is None:
                # SlotSet("drinks",add_drinks)
                # SlotSet("drinks_num",add_drinks_num)
                return [SlotSet("food",add_food),SlotSet("food_num",add_food_num),
                        SlotSet("drinks",add_drinks),SlotSet("drinks_num",add_drinks_num)]

            # drinks is a single value
            elif isinstance(drinks,str): 
                add_drinks.append(drinks)
                add_drinks_num.append(drinks_num)

                print(f"add_drinks: {add_drinks}")
                print(f"add_drinks_num: {add_drinks_num}")

                # SlotSet("drinks",add_drinks)
                # SlotSet("drinks_num",add_drinks_num)
                return [SlotSet("food",add_food),SlotSet("food_num",add_food_num),
                        SlotSet("drinks",add_drinks),SlotSet("drinks_num",add_drinks_num)]


            # drinks is a list
            else: 
                for indexx in range(len(add_drinks)):
                    drinks.append(add_drinks[indexx]) # modify drinks list
                    drinks_num.append(add_drinks_num[indexx]) # modify drinks_num list

                print(f"change_drinks: {drinks}")
                print(f"change_drinks_num: {drinks_num}")

                # SlotSet("drinks",drinks)
                # SlotSet("drinks_num",drinks_num)
                return [SlotSet("food",add_food),SlotSet("food_num",add_food_num),
                        SlotSet("drinks",drinks),SlotSet("drinks_num",drinks_num)]



        # food is a list ####################################################################

        else: 
            for index in range(len(add_food)):
                food.append(add_food[index]) # modify food list
                food_num.append(add_food_num[index]) # modify food_num list

            print(f"change_food: {food}")
            print(f"change_food_num: {food_num}")

            # SlotSet("food",food)
            # SlotSet("food_num",food_num)

            # modify drinks slot

            # drinks is None
            if drinks is None:
                # SlotSet("drinks",add_drinks)
                # SlotSet("drinks_num",add_drinks_num)
                return [SlotSet("food",food),SlotSet("food_num",food_num),
                        SlotSet("drinks",add_drinks),SlotSet("drinks_num",add_drinks_num)]


            # drinks is a single value
            elif isinstance(drinks,str): 
                add_drinks.append(drinks)
                add_drinks_num.append(drinks_num)

                print(f"add_drinks: {add_drinks}")
                print(f"add_drinks_num: {add_drinks_num}")

                # SlotSet("drinks",add_drinks)
                # SlotSet("drinks_num",add_drinks_num)
                return [SlotSet("food",food),SlotSet("food_num",food_num),
                        SlotSet("drinks",add_drinks),SlotSet("drinks_num",add_drinks_num)]


            # drinks is a list
            else: 
                for indexx in range(len(add_drinks)):
                    drinks.append(add_drinks[indexx]) # modify drinks list
                    drinks_num.append(add_drinks_num[indexx]) # modify drinks_num list

                print(f"change_drinks: {drinks}")
                print(f"change_drinks_num: {drinks_num}")

                # SlotSet("drinks",drinks)
                # SlotSet("drinks_num",drinks_num)
                return [SlotSet("food",food),SlotSet("food_num",food_num),
                        SlotSet("drinks",drinks),SlotSet("drinks_num",drinks_num)]








class changeOrder(Action):

    def name(self) -> Text:
        return "action_change_order"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain:Dict[Text, Any])-> List[Dict[Text, Any]]:

        food_list = ["big mac", "cheeseburger", "bacon angus", "classic angus", "mcspicy",
              "chicken salad", "garden salad", "chicken nuggets", "fries", "apple pie", "hash brown"]

        drinks_list = ["strawberry sundae", "coke", "sprite", "ice latte", "apple juice"]

        food = tracker.get_slot("food")
        food_num = tracker.get_slot("food_num")

        drinks = tracker.get_slot("drinks")
        drinks_num = tracker.get_slot("drinks_num")

        change = tracker.get_slot("change")
        change_num = tracker.get_slot("change_num")

        change_food = []
        change_food_num = []

        change_drinks = []
        change_drinks_num = []

        if isinstance(change,list): # change is a list
            for num in range(len(change)):
                if change[num].lower() in food_list:
                    change_food.append(change[num].lower())
                    change_food_num.append(change_num[num])

                else:
                    change_drinks.append(change[num].lower())
                    change_drinks_num.append(change_num[num])

        else: # change is a single value
            if change.lower() in food_list:
                change_food.append(change.lower())
                change_food_num.append(change_num)

            else:
                change_drinks.append(change.lower())
                change_drinks_num.append(change_num)

        print(f"change_food: {change_food}")
        print(f"change_food_num: {change_food_num}")
        print(f"change_drinks: {change_drinks}")
        print(f"change_drinks_num: {change_drinks_num}")

        # modify food slot

        # food is a single value ####################################################################

        if isinstance(food,str): 
            # convert food value into lowercase
            food = food.lower()
            change_food_list = []
            change_food_num_list = []

            # find the matched items in food and remove items in change_food[]
            for index in range(len(change_food)):
                # find the matched item 
                if change_food[index] != food:
                    # add items into change_food_list[]
                    change_food_list.append(change_food[index])
                    change_food_num_list.append(change_food_num[index])


            # then assign change_food_list[] to food
            food = change_food_list
            food_num = change_food_num_list


            print(f"change_food: {food}")
            print(f"change_food_num: {food_num}")

            # SlotSet("food",food)
            # SlotSet("food_num",food_num)


            # modify drinks slot

            # drinks is a single value
            if isinstance(drinks,str): 
                # convert drinks value into lowercase
                drinks = drinks.lower()
                change_drinks_list = []
                change_drinks_num_list = []

                # find the matched items in drinks and remove items in change_drinks[]
                for index1 in range(len(change_drinks)):
                    # find the matched item 
                    if change_drinks[index1] != drinks:
                        # add items into changed_drinks_list[]
                        change_drinks_list.append(change_drinks[index1])
                        change_drinks_num_list.append(change_drinks_num[index1])


                # then assign change_drinks to drinks
                drinks = change_drinks_list
                drinks_num = change_drinks_num_list

                print(f"change_drinks: {drinks}")
                print(f"change_drinks_num: {drinks_num}")

                # SlotSet("drinks",drinks)
                # SlotSet("drinks_num",drinks_num)
                return [SlotSet("food",food),SlotSet("food_num",food_num),
                        SlotSet("drinks",drinks),SlotSet("drinks_num",drinks_num)]

            # drinks is a list
            else: 
                # convert all drinks value into lowercase
                drinks_low = []
                for d in drinks:
                    drinks_low.append(d.lower())

                # find the matched items in drinks_low[] and remove items
                for index2 in range(len(change_drinks)):
                    # find the matched item
                    if change_drinks[index2] in drinks_low:
                        # remove the item in drinks_low[]
                        num2 = drinks_low.index(change_drinks[index2])
                        drinks_low.pop(num2)
                        drinks_num.pop(num2)

                    else:
                        # add items into drinks_low
                        drinks_low.append(change_drinks[index2])
                        drinks_num.append(change_drinks_num[index2])



                print(f"change_drinks: {drinks_low}")
                print(f"change_drinks_num: {drinks_num}")

                # SlotSet("drinks",drinks_low)
                # SlotSet("drinks_num",drinks_num)
                return [SlotSet("food",food),SlotSet("food_num",food_num),
                        SlotSet("drinks",drinks_low),SlotSet("drinks_num",drinks_num)]





        # food is a list ####################################################################

        else:
            # convert all food value into lowercase
            food_low = []
            for f in food:
                food_low.append(f.lower())

            # find the matched items in food_low[] and remove items
            for index in range(len(change_food)):
                # find the matched item
                if change_food[index] in food_low:
                    # remove the item in food_low[]
                    num = food_low.index(change_food[index])
                    food_low.pop(num)
                    food_num.pop(num)

                else:
                    # add items into food_low[]
                    food_low.append(change_food[index])
                    food_num.append(change_food_num[index])



            print(f"change_food: {food_low}")
            print(f"change_food_num: {food_num}")

            # SlotSet("food",food_low)
            # SlotSet("food_num",food_num)




            # modify drinks slot

            # drinks is a single value
            if isinstance(drinks,str): 
                # convert drinks value into lowercase
                drinks = drinks.lower()
                change_drinks_list = []
                change_drinks_num_list = []

                # find the matched items in drinks and remove items in change_drinks[]
                for index1 in range(len(change_drinks)):
                    # find the matched item 
                    if change_drinks[index1] != drinks:
                        # add items into changed_drinks_list[]
                        change_drinks_list.append(change_drinks[index1])
                        change_drinks_num_list.append(change_drinks_num[index1])


                # then assign change_drinks to drinks
                drinks = change_drinks_list
                drinks_num = change_drinks_num_list

                print(f"change_drinks: {drinks}")
                print(f"change_drinks_num: {drinks_num}")

                # SlotSet("drinks",drinks)
                # SlotSet("drinks_num",drinks_num)
                return [SlotSet("food",food_low),SlotSet("food_num",food_num),
                        SlotSet("drinks",drinks),SlotSet("drinks_num",drinks_num)]

            # drinks is a list
            else: 
                # convert all drinks value into lowercase
                drinks_low = []
                for d in drinks:
                    drinks_low.append(d.lower())

                # find the matched items in drinks_low[] and remove items
                for index2 in range(len(change_drinks)):
                    # find the matched item
                    if change_drinks[index2] in drinks_low:
                        # remove the item in drinks_low[]
                        num2 = drinks_low.index(change_drinks[index2])
                        drinks_low.pop(num2)
                        drinks_num.pop(num2)

                    else:
                        # add items into drinks_low
                        drinks_low.append(change_drinks[index2])
                        drinks_num.append(change_drinks_num[index2])



                print(f"change_drinks: {drinks_low}")
                print(f"change_drinks_num: {drinks_num}")

                # SlotSet("drinks",drinks_low)
                # SlotSet("drinks_num",drinks_num)
                return [SlotSet("food",food_low),SlotSet("food_num",food_num),
                        SlotSet("drinks",drinks_low),SlotSet("drinks_num",drinks_num)]






     

class removeOrder(Action):

    def name(self) -> Text:
        return "action_remove_order"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain:Dict[Text, Any])-> List[Dict[Text, Any]]:

        food_list = ["big mac", "cheeseburger", "bacon angus", "classic angus", "mcspicy",
              "chicken salad", "garden salad", "chicken nuggets", "fries", "apple pie", "hash brown"]

        drinks_list = ["strawberry sundae", "coke", "sprite", "ice latte", "apple juice"]

        food = tracker.get_slot("food")
        food_num = tracker.get_slot("food_num")

        drinks = tracker.get_slot("drinks")
        drinks_num = tracker.get_slot("drinks_num")

        remove = tracker.get_slot("remove")
        remove_num = tracker.get_slot("remove_num")

        remove_food = []
        remove_food_num = []

        remove_drinks = []
        remove_drinks_num = []

        if isinstance(remove,list): # remove is a list
            for num in range(len(remove)):
                if remove[num].lower() in food_list:
                    remove_food.append(remove[num].lower())
                    remove_food_num.append(remove_num[num])

                else:
                    remove_drinks.append(remove[num].lower())
                    remove_drinks_num.append(remove_num[num])

        else: # remove is a single value
            if remove.lower() in food_list:
                remove_food.append(remove.lower())
                remove_food_num.append(remove_num)

            else:
                remove_drinks.append(remove.lower())
                remove_drinks_num.append(remove_num)



        # modify food slot

        # food is a single value ####################################################################

        if isinstance(food,str): 
            food = None
            food_num = None

            print(f"remove_food: {food}")
            print(f"remove_food_num: {food_num}")

            # SlotSet("food",food)
            # SlotSet("food_num",food_num)


            # modify drinks slot

            # drinks is a single value
            if isinstance(drinks,str): 
                drinks = None
                drinks_num = None

                print(f"remove_drinks: {drinks}")
                print(f"remove_drinks_num: {drinks_num}")

                # SlotSet("drinks",drinks)
                # SlotSet("drinks_num",drinks_num)
                return [SlotSet("food",food),SlotSet("food_num",food_num),
                        SlotSet("drinks",drinks),SlotSet("drinks_num",drinks_num)]

            # drinks is a list
            else: 
                # convert all drinks value into lowercase
                drinks_low = []
                for d in drinks:
                    drinks_low.append(d.lower())

                # find the matched items in drinks_low[] and remove items
                for index in range(len(remove_drinks)):
                    # find the matched item
                    if remove_drinks[index] in drinks_low:
                        num = drinks_low.index(remove_drinks[index])
                        # then remove the item
                        drinks_low.pop(num)
                        drinks_num.pop(num)


                print(f"remove_drinks: {drinks_low}")
                print(f"remove_drinks_num: {drinks_num}")

                # SlotSet("drinks",drinks_low)
                # SlotSet("drinks_num",drinks_num)
                return [SlotSet("food",food),SlotSet("food_num",food_num),
                        SlotSet("drinks",drinks_low),SlotSet("drinks_num",drinks_num)]



        # food is a list ####################################################################

        else: 
            # convert all food value into lowercase
            food_low = []
            for f in food:
                food_low.append(f.lower())

            # find the matched items in food_low[] and remove items
            for indexx in range(len(remove_food)):
                # find the matched item 
                if remove_food[indexx] in food_low:
                    num1 = food_low.index(remove_food[indexx])
                    # then delete the item
                    food_low.pop(num1)
                    food_num.pop(num1)


            print(f"remove_food: {food_low}")
            print(f"remove_food_num: {food_num}")

            # SlotSet("food",food_low)
            # SlotSet("food_num",food_num)


            # modify drinks slot

            # drinks is a single value
            if isinstance(drinks,str):
                drinks = None
                drinks_num = None

                print(f"remove_drinks: {drinks}")
                print(f"remove_drinks_num: {drinks_num}")

                # SlotSet("drinks",drinks)
                # SlotSet("drinks_num",drinks_num)
                return [SlotSet("food",food_low),SlotSet("food_num",food_num),
                        SlotSet("drinks",drinks),SlotSet("drinks_num",drinks_num)]

            
            # drinks is a list
            else: 
                # convert all drinks value into lowercase
                drinks_low = []
                for dr in drinks:
                    drinks_low.append(dr.lower())

                # find the matched items in drinks_low[] and remove items
                for indexxx in range(len(remove_drinks)):
                    # find the matched item
                    if remove_drinks[indexxx] in drinks_low:
                        num2 = drinks_low.index(remove_drinks[indexxx])
                        # then remove the item
                        drinks_low.pop(num2)
                        drinks_num.pop(num2)


                print(f"remove_drinks: {drinks_low}")
                print(f"remove_drinks_num: {drinks_num}")

                # SlotSet("drinks",drinks_low)
                # SlotSet("drinks_num",drinks_num)
                return [SlotSet("food",food_low),SlotSet("food_num",food_num),
                        SlotSet("drinks",drinks_low),SlotSet("drinks_num",drinks_num)]
