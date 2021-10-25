import json,copy
import random
# train_data = json.load(open('data/train.json'))
dele_data = json.load(open('/data/xyzhang/RL_research_folder/soloist-rl/soloist-cuhk/examples/multiwoz/data/standard/train/train5.train.delex.json'))
# dialog_acts = json.load(open("/home/xiaoying/soloist-cuhk/examples/multiwoz/data/multi-woz/dialogue_acts.json"))
# input_no_delexicalize_data = json.load(open('/home/xiaoying/soloist-cuhk/examples/multiwoz/data/standard/restaurant/restaurant5.train_data.json'))
db_file = json.load(open('db/train_db.json'))
# 110
# special_tokens_res = open('/home/xiaoying/soloist-cuhk/examples/multiwoz/resource/special_tokens_restaurant.txt')
# valid_resnames = json.load(open('/home/xiaoying/soloist-cuhk/examples/multiwoz/data/standard/restaurant/restaurant45.valid_resnames.json'))
time_db = json.load(open('data/standard/hotel/hotel_stay_db.json'))
people_db = json.load(open('data/standard/hotel/hotel_people_db.json'))
# day_values = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
requestables = ['phone', 'address', 'postcode', 'reference', 'id']
# random.choice()
domain = 'train'
def write_data_for_internal_v1(dele_data, output_file, time_values, people_values, db_file, domain):

    # day_values = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    examples = {}
    idxs = list(dele_data.keys())
    print(idxs)
    for i, res_db in enumerate(db_file):
        ids = random.sample(idxs, 1)
        print(ids)
        for j, filename in enumerate(ids):
            origin_file = dele_data[filename]
            print(filename)
            # for filename, info in origin_file.items():
            info = origin_file
            filename = filename.upper().split('.')[0]
            # print(filename)
            info1 = copy.deepcopy(info)
            filename1 = filename + '_' + str(i) + '_' + str(j)
            fname1 = filename1 + '.json'
            print(fname1)
            # stay = random.choice(time_values)
            # people = random.choice(people_values)
            # day = random.choice(day_values)

            # time_people_day = {
            #     # "time": time,
            #     # "hotel_num": hotel_num,
            #     "stay": stay,
            #     "people": people,
            #     "day": day
            # }

            if info1["goal"][domain]:
                if "info" in info1["goal"][domain].keys():
                    for item in info1["goal"][domain]["info"].keys():
                        if info1["goal"][domain]["info"][item] != "dontcare" and info1["goal"][domain]["info"][item] != "not mentioned":
                            info1["goal"][domain]["info"][item] = res_db[item]
                if domain in info1["goal"][domain].keys():
                    info1["goal"][domain][domain] = res_db["name"]
# [train_price]
# [train_reference]
# [train_id]
# [value_count]
# [value_day]
# [value_place]
# [value_time]
# [value_price]
                turns = info1["log"]
                for turn in turns:
                    print(turn["text"])
                    turn["text"] = turn["text"].replace("departs [value_place] on [value_day] at [value_time]", "departs [value_place] on [value_day] at " + res_db["leaveAt"])
                    turn["text"] = turn["text"].replace("leave after [value_time]", "leave after " + res_db["leaveAt"])
                    turn["text"] = turn["text"].replace("departs after [value_time]", "departs after " + res_db["leaveAt"])
                    turn["text"] = turn["text"].replace("departs [value_day] at [value_time]", "departs [value_day] at " + res_db["leaveAt"])
                    turn["text"] = turn["text"].replace("arriving at [value_place] on [value_day] at [value_time]", "arriving at [value_place] on [value_day] at " + res_db["arriveBy"])
                    turn["text"] = turn["text"].replace("arrives at [value_place] [value_day] at [value_time]", "arrives at [value_place] [value_day] at " + res_db["arriveBy"])
                    turn["text"] = turn["text"].replace("[" + domain + "_id]", res_db["trainID"])
                    turn["text"] = turn["text"].replace("[" + domain + "_price]", res_db["price"])
                    # if "reference" in res_db.keys():
                    #     turn["text"] = turn["text"].replace("[" + domain +"_reference]", res_db["reference"])
                    turn["text"] = turn["text"].replace("[value_day]", res_db["day"])
                    turn["text"] = turn["text"].replace("departs [value_place]", "departs " + res_db["departure"])
                    turn["text"] = turn["text"].replace("from [value_place]", "from " + res_db["departure"])
                    turn["text"] = turn["text"].replace("leaving [value_place]", "leaving " + res_db["departure"])
                    turn["text"] = turn["text"].replace("leave [value_place]", "leave " + res_db["departure"])
                    turn["text"] = turn["text"].replace("to [value_place]", "to " + res_db["destination"])
                    turn["text"] = turn["text"].replace("to the [value_place]", "to the " + res_db["destination"])
                    turn["text"] = turn["text"].replace("for [value_place]", "for " + res_db["destination"])
                    turn["text"] = turn["text"].replace("leaves at [value_time]", "leaves at " + res_db["leaveAt"])
                    turn["text"] = turn["text"].replace("leave at [value_time]", "leave at " + res_db["leaveAt"])
                    turn["text"] = turn["text"].replace("[value_count] minutes", res_db["duration"] + " minutes")
                    turn["text"] = turn["text"].replace("by [value_time]", "by " + res_db["arriveBy"])
                    turn["text"] = turn["text"].replace("arrive at [value_time]", "arrive at " + res_db["arriveBy"])
                    turn["text"] = turn["text"].replace("arrives at [value_time]", "arrives at " + res_db["arriveBy"])
                    turn["text"] = turn["text"].replace("at [value_place]", "at " + res_db["destination"])
                    turn["text"] = turn["text"].replace("[value_price]", res_db["price"])
                    # turn["text"] = turn["text"].replace("[value_area]", res_db["area"])
                    print(turn["text"])
                    if domain in turn['metadata'].keys():
                        if "book" in turn['metadata'][domain].keys():
                            # for item, value in time_people_day.items():
                            #     if item in turn['metadata'][domain]["book"].keys():
                            #         if turn['metadata'][domain]["book"][item] != "": 
                            #             turn['metadata'][domain]["book"][item] = value
                            if "booked" in turn['metadata'][domain]["book"]:
                                if turn['metadata'][domain]["book"]["booked"] != []:
                                    for item in turn['metadata'][domain]["book"]["booked"][0].keys():
                                        if item in res_db:
                                            turn['metadata'][domain]["book"]["booked"][0][item] = res_db[item]

                        if "semi" in turn['metadata'][domain].keys():  
                            for item in turn['metadata'][domain]["semi"].keys():
                                if turn['metadata'][domain]["semi"][item] != "dontcare" and turn['metadata'][domain]["semi"][item] != "not mentioned" and turn['metadata'][domain]["semi"][item] != "" and turn['metadata'][domain]["semi"][item] != "don't care" and turn['metadata'][domain]["semi"][item] != "dont care" and turn['metadata'][domain]["semi"][item] != "do n't care":
                                    turn['metadata'][domain]["semi"][item] = res_db[item]

            examples[fname1] = info1
    print(len(examples))
    json.dump(examples, open(output_file,'w'),indent=2)
write_data_for_internal_v1(dele_data, '/data/xyzhang/RL_research_folder/soloist-rl/soloist-cuhk/examples/multiwoz/data/standard/train/train5_constructed.train.json', time_db, people_db, db_file, domain)

# write_data_for_internal_v1(full_data, '/home/xiaoying/soloist-cuhk/examples/multiwoz/data/standard/restaurant/restaurant45.valid_data.json', input_idx_data, '/home/xiaoying/soloist-cuhk/examples/multiwoz/data/standard/restaurant/restaurant45.valid_resnames.json')
# write_data_for_internal_v1(test_data, 'test.soloist.json', 'test.idx.json')