print("This grade calculator can determine 1) your current average or 2) the minimum average you need to reach a target grade.")

calc_choice = input("What would you like the grade calculator to do for you? \n")

while(calc_choice != "1" and calc_choice != "2"):
    print("Sorry, you have to type either 1 or 2")
    print("This grade calculator can determine 1) your current average or 2) the minimum average you need to reach a target grade.")
    calc_choice = input("What would you like the grade calculator to do for you? \n")

if(calc_choice == "1"):
    number_mp = 0
    print("Testing 1")
    mp1 = float(input("What did you get for the 1st marking period? \n"))
    mp2 = float(input("What did you get for the 2nd marking period? \n"))
    mp3 = float(input("What did you get for the 3rd marking period? \n"))
    mp4 = float(input("What did you get for the 4th marking period? \n"))
    avg = (mp1 + mp2 + mp3 + mp4) / 4
    print("Your average was " + str(round(avg, 2)))

else:
    grade_target = float(input("What grade would you like to get at the end of the year? \n"))
    mp1 = float(input("What did you get for the 1st marking period? \n"))
    mp2 = float(input("What did you get for the 2nd marking period? \n"))
    mp3 = float(input("What did you get for the 3rd marking period? \n"))
    grade_needed = (grade_target * 4) - (mp1 + mp2 + mp3)    
    print("You will need a minimum average of " +str(round(grade_needed, 2)))
