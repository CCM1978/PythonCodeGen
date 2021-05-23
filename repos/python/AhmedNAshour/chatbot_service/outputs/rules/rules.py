def findDecision(obj): #obj[0]: Fever, obj[1]: Tiredness, obj[2]: Dry-Cough, obj[3]: Difficulty-in-Breathing, obj[4]: Sore-Throat, obj[5]: Pains, obj[6]: Nasal-Congestion, obj[7]: Runny-Nose, obj[8]: Diarrhea, obj[9]: Smell and taste loss
   # {"feature": "Smell and taste loss", "instances": 505968, "metric_value": 0.9536, "depth": 1}
   if obj[9]>0:
      return 'yes'
   elif obj[9]<=0:
      return 'no'
   else:
      return 'no'
