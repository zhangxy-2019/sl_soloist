import json,copy
import random
# train_data = json.load(open('data/train.json'))
dele_data = json.load(open('/data/xyzhang/RL_research_folder/soloist-rl/soloist-cuhk/examples/multiwoz/data/standard/new_restaurant/restaurant_delivery30simplified.test.delex.json'))
# input_idx_data = json.load(open('/data/xyzhang/RL_research_folder/soloist-rl/soloist-cuhk/examples/multiwoz/data/standard/new_restaurant/restaurant_domain_extension5.idx.json'))
# dialog_acts = json.load(open("/home/xiaoying/soloist-cuhk/examples/multiwoz/data/multi-woz/dialogue_acts.json"))
# input_no_delexicalize_data = json.load(open('/home/xiaoying/soloist-cuhk/examples/multiwoz/data/standard/restaurant/restaurant5.train_data.json'))
db_file = json.load(open('db/restaurant_db.json'))
time_db = json.load(open('db/restaurant_time_db.json'))
people_db = json.load(open('db/restaurant_people_db.json'))
day_values = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
# requestables = ['phone', 'address', 'postcode', 'reference', 'id']
# random.choice()
def write_data_replace_slots(input_delex, output_file, time_values, people_values, db_file):
    # file_list = list(set(input_idx_data))

    day_values = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    examples = {}
    # num_new_examples = len(db_file) * 2
    idxs = list(range(len(db_file)))
    # print(idxs)
    for fname, fcontent in input_delex.items():
        # origin_file = fcontent
        filename = fname.upper().split('.')[0]
        i = random.choice(idxs)
        res_db = db_file[i]
        info1 = copy.deepcopy(fcontent)
        fname1 = filename + '_' + str(i) + '.json'
        time = random.choice(time_values)
        people = random.choice(people_values)
        day = random.choice(day_values)
        time_people_day = {
            "time": time,
            "people": people,
            "day": day
        }

        if info1["goal"]["restaurant"]:
            if "info" in info1["goal"]["restaurant"].keys():
                for item in info1["goal"]["restaurant"]["info"].keys():
                    if "dontcare" not in info1["goal"]["restaurant"]["info"][item] and "not mentioned" not in info1["goal"]["restaurant"]["info"][item]:
                        info1["goal"]["restaurant"]["info"][item] = res_db[item]

            if "book" in info1["goal"]["restaurant"].keys():
                if "invalid" in info1["goal"]["restaurant"]["book"].keys():
                    if not info1["goal"]["restaurant"]["book"]["invalid"]:
                        for item, value in time_people_day.items():
                            if item in info1["goal"]["restaurant"]["book"].keys(): 
                                if info1["goal"]["restaurant"]["book"][item] != "": 
                                    info1["goal"]["restaurant"]["book"][item] = value
        
            turns = info1["log"]
            for turn in turns:
                print(turn["text"])
                turn["text"] = turn["text"].replace("[restaurant_address]", res_db["address"])
                turn["text"] = turn["text"].replace("[restaurant_area]", res_db["area"])
                turn["text"] = turn["text"].replace("[restaurant_food]", res_db["food"])
                turn["text"] = turn["text"].replace("[restaurant_name]", res_db["name"])
                if "phone" in res_db.keys():
                    turn["text"] = turn["text"].replace("[restaurant_phone]", res_db["phone"])
                turn["text"] = turn["text"].replace("[restaurant_postcode]", res_db["postcode"])
                turn["text"] = turn["text"].replace("[restaurant_pricerange]", res_db["pricerange"])
                if "reference" in res_db.keys():
                    turn["text"] = turn["text"].replace("[restaurant_reference]", res_db["reference"])
                turn["text"] = turn["text"].replace("[value_day]", day)
                turn["text"] = turn["text"].replace("[value_food]", res_db["food"])
                turn["text"] = turn["text"].replace("[value_count]", people)
                turn["text"] = turn["text"].replace("[value_pricerange]", res_db["pricerange"])
                turn["text"] = turn["text"].replace("[value_time]", time)
                turn["text"] = turn["text"].replace("[value_area]", res_db["area"])
                print(turn["text"])
                if 'restaurant' in turn['metadata'].keys():
                    if "book" in turn['metadata']['restaurant'].keys():
                        for item, value in time_people_day.items():
                            if item in turn['metadata']['restaurant']["book"].keys():
                                if turn['metadata']['restaurant']["book"][item] != "": 
                                    turn['metadata']['restaurant']["book"][item] = value
                        if "booked" in turn['metadata']['restaurant']["book"]:
                            if turn['metadata']['restaurant']["book"]["booked"] != []:
                                for item in turn['metadata']['restaurant']["book"]["booked"][0].keys():
                                    if item in res_db:
                                        turn['metadata']['restaurant']["book"]["booked"][0][item] = res_db[item]

                    if "semi" in turn['metadata']['restaurant'].keys():  
                        for item in turn['metadata']['restaurant']["semi"].keys():
                            if "dontcare" not in turn['metadata']['restaurant']["semi"][item] and "not mentioned" not in turn['metadata']['restaurant']["semi"][item] and turn['metadata']['restaurant']["semi"][item] != "":
                                turn['metadata']['restaurant']["semi"][item] = res_db[item]

        examples[fname1] = info1
    print(len(examples))
    json.dump(examples, open(output_file,'w'),indent=2)
# write_data_for_internal_v1(dele_data, '/data/xyzhang/RL_research_folder/soloist-rl/soloist-cuhk/examples/multiwoz/data/standard/new_restaurant/restaurant_domain_extension5_augmented.full_data.json',input_idx_data, time_db, people_db, db_file)
write_data_replace_slots(dele_data, '/data/xyzhang/RL_research_folder/soloist-rl/soloist-cuhk/examples/multiwoz/data/standard/new_restaurant/restaurant_delivery30simplified.test.data.json', time_db, people_db, db_file)

def write_data_for_internal_v1(input_file, output_file, time_values, people_values, db_file):
    file_list = list(set(input_idx_data))

    day_values = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    examples = {}
    for i, res_db in enumerate(db_file):
        ids = random.sample(idxs, 2)
        print(ids)
        for j, filename in enumerate(ids):
            origin_file = dele_data[filename]
            # for filename, info in origin_file.items():
            info = origin_file
            filename = filename.upper().split('.')[0]
            info1 = copy.deepcopy(info)
            fname1 = filename + '_' + str(i) + '_' + str(j) + '.json'
            time = random.choice(time_values)
            people = random.choice(people_values)
            day = random.choice(day_values)
            time_people_day = {
                "time": time,
                "people": people,
                "day": day
            }

            if info1["goal"]["restaurant"]:
                if "info" in info1["goal"]["restaurant"].keys():
                    for item in info1["goal"]["restaurant"]["info"].keys():
                        if "dontcare" not in info1["goal"]["restaurant"]["info"][item] and "not mentioned" not in info1["goal"]["restaurant"]["info"][item]:
                            info1["goal"]["restaurant"]["info"][item] = res_db[item]

                if "book" in info1["goal"]["restaurant"].keys():
                    if "invalid" in info1["goal"]["restaurant"]["book"].keys():
                        if not info1["goal"]["restaurant"]["book"]["invalid"]:
                            for item, value in time_people_day.items():
                                if item in info1["goal"]["restaurant"]["book"].keys(): 
                                    if info1["goal"]["restaurant"]["book"][item] != "": 
                                        info1["goal"]["restaurant"]["book"][item] = value
            
                turns = info1["log"]
                for turn in turns:
                    print(turn["text"])
                    turn["text"] = turn["text"].replace("[restaurant_address]", res_db["address"])
                    turn["text"] = turn["text"].replace("[restaurant_area]", res_db["area"])
                    turn["text"] = turn["text"].replace("[restaurant_food]", res_db["food"])
                    turn["text"] = turn["text"].replace("[restaurant_name]", res_db["name"])
                    if "phone" in res_db.keys():
                        turn["text"] = turn["text"].replace("[restaurant_phone]", res_db["phone"])
                    turn["text"] = turn["text"].replace("[restaurant_postcode]", res_db["postcode"])
                    turn["text"] = turn["text"].replace("[restaurant_pricerange]", res_db["pricerange"])
                    if "reference" in res_db.keys():
                        turn["text"] = turn["text"].replace("[restaurant_reference]", res_db["reference"])
                    turn["text"] = turn["text"].replace("[value_day]", day)
                    turn["text"] = turn["text"].replace("[value_food]", res_db["food"])
                    turn["text"] = turn["text"].replace("[value_count]", people)
                    turn["text"] = turn["text"].replace("[value_pricerange]", res_db["pricerange"])
                    turn["text"] = turn["text"].replace("[value_time]", time)
                    turn["text"] = turn["text"].replace("[value_area]", res_db["area"])
                    print(turn["text"])
                    if 'restaurant' in turn['metadata'].keys():
                        if "book" in turn['metadata']['restaurant'].keys():
                            for item, value in time_people_day.items():
                                if item in turn['metadata']['restaurant']["book"].keys():
                                    if turn['metadata']['restaurant']["book"][item] != "": 
                                        turn['metadata']['restaurant']["book"][item] = value
                            if "booked" in turn['metadata']['restaurant']["book"]:
                                if turn['metadata']['restaurant']["book"]["booked"] != []:
                                    for item in turn['metadata']['restaurant']["book"]["booked"][0].keys():
                                        if item in res_db:
                                            turn['metadata']['restaurant']["book"]["booked"][0][item] = res_db[item]

                        if "semi" in turn['metadata']['restaurant'].keys():  
                            for item in turn['metadata']['restaurant']["semi"].keys():
                                if "dontcare" not in turn['metadata']['restaurant']["semi"][item] and "not mentioned" not in turn['metadata']['restaurant']["semi"][item] and turn['metadata']['restaurant']["semi"][item] != "":
                                    turn['metadata']['restaurant']["semi"][item] = res_db[item]

            examples[fname1] = info1
    print(len(examples))
    json.dump(examples, open(output_file,'w'),indent=2)

# write_data_replace_slots(dele_data, '/data/xyzhang/RL_research_folder/soloist-rl/soloist-cuhk/examples/multiwoz/data/standard/new_restaurant/restaurant_deliverynewtest15.data.json', time_db, people_db, db_file)