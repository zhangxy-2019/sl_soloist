import json,copy
import random
# train_data = json.load(open('data/train.json'))
dele_data = json.load(open('/data/xyzhang/RL_research_folder/soloist-rl/soloist-cuhk/examples/multiwoz/data/standard/attraction/attraction5.train.delex_for_construced.json'))
db_file = json.load(open('db/attraction_db.json'))
# # SNG1129 museum
# SNG1173 college, specific attraction, can by replaced by any one
# SNG1061 nightclub 
# SNG1072 architecture
# SNG1148.json
# random.choice()
domain = 'attraction'
def write_data_for_internal_v1(dele_data, output_file, domain):

    # day_values = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    examples = {}
    idxs = list(dele_data.keys())
    print(idxs)
    # # SNG1173
    # idxs = idxs.remove("SNG1173.json")
    origin_file1173 = dele_data["SNG1173.json"] #
    for i, res_db in enumerate(db_file):
        filename1 = "SNG1173.json".upper().split('.')[0] + '_' + str(i)
        fname1 = filename1 + '.json'
        print(fname1)
        info1 = copy.deepcopy(origin_file1173)

        if info1["goal"][domain]:
            if "info" in info1["goal"][domain].keys():
                for item in info1["goal"][domain]["info"].keys():
                    if info1["goal"][domain]["info"][item] != "dontcare" and info1["goal"][domain]["info"][item] != "not mentioned":
                        info1["goal"][domain]["info"][item] = res_db[item]
            if domain in info1["goal"][domain].keys():
                info1["goal"][domain][domain] = res_db["name"]
            
            turns = info1["log"]
            for turn in turns:
                print(turn["text"])
                turn["text"] = turn["text"].replace("[" + domain + "_address]", res_db["address"])
                turn["text"] = turn["text"].replace("[" + domain + "_name]", res_db["name"])
                turn["text"] = turn["text"].replace("[" + domain + "_phone]", res_db["phone"])
                turn["text"] = turn["text"].replace("[" + domain + "_postcode]", res_db["postcode"])
                turn["text"] = turn["text"].replace("[value_area]", res_db["area"])
                turn["text"] = turn["text"].replace("[attraction_type]", res_db["type"])
                print(turn["text"])
                if domain in turn['metadata'].keys():
                    if "book" in turn['metadata'][domain].keys():
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

    origin_file1107 = dele_data["SNG1107.json"] # res_db["entrance fee"] == "free" #
    for i, res_db in enumerate(db_file):
        if res_db["entrance fee"] == "free":
            filename1 = "SNG1107.json".upper().split('.')[0] + '_' + str(i)
            fname1 = filename1 + '.json'
            print(fname1)
            info1 = copy.deepcopy(origin_file1107)
            if info1["goal"][domain]:
                if "info" in info1["goal"][domain].keys():
                    for item in info1["goal"][domain]["info"].keys():
                        if info1["goal"][domain]["info"][item] != "dontcare" and info1["goal"][domain]["info"][item] != "not mentioned":
                            info1["goal"][domain]["info"][item] = res_db[item]
                if domain in info1["goal"][domain].keys():
                    info1["goal"][domain][domain] = res_db["name"]
                
                turns = info1["log"]
                for turn in turns:
                    print(turn["text"])
                    turn["text"] = turn["text"].replace("[" + domain + "_address]", res_db["address"])
                    turn["text"] = turn["text"].replace("[" + domain + "_name]", res_db["name"])
                    turn["text"] = turn["text"].replace("[" + domain + "_phone]", res_db["phone"])
                    turn["text"] = turn["text"].replace("[" + domain + "_postcode]", res_db["postcode"])
                    turn["text"] = turn["text"].replace("[value_area]", res_db["area"])
                    turn["text"] = turn["text"].replace("[attraction_type]", res_db["type"])
                    print(turn["text"])
                    if domain in turn['metadata'].keys():
                        if "book" in turn['metadata'][domain].keys():
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


    origin_file1397 = dele_data["SNG1397.json"] # res_db["entrance fee"] == "free" #
    for i, res_db in enumerate(db_file):
        if res_db["entrance fee"] == "free":
            filename1 = "SNG1397.json".upper().split('.')[0] + '_' + str(i)
            fname1 = filename1 + '.json'
            print(fname1)
            info1 = copy.deepcopy(origin_file1397)
            if info1["goal"][domain]:
                if "info" in info1["goal"][domain].keys():
                    for item in info1["goal"][domain]["info"].keys():
                        if info1["goal"][domain]["info"][item] != "dontcare" and info1["goal"][domain]["info"][item] != "not mentioned":
                            info1["goal"][domain]["info"][item] = res_db[item]
                if domain in info1["goal"][domain].keys():
                    info1["goal"][domain][domain] = res_db["name"]
                
                turns = info1["log"]
                for turn in turns:
                    print(turn["text"])
                    turn["text"] = turn["text"].replace("[" + domain + "_address]", res_db["address"])
                    turn["text"] = turn["text"].replace("[" + domain + "_name]", res_db["name"])
                    turn["text"] = turn["text"].replace("[" + domain + "_phone]", res_db["phone"])
                    turn["text"] = turn["text"].replace("[" + domain + "_postcode]", res_db["postcode"])
                    turn["text"] = turn["text"].replace("[value_area]", res_db["area"])
                    turn["text"] = turn["text"].replace("[value_count] pounds", res_db["entrance fee"])                    
                    turn["text"] = turn["text"].replace("[attraction_type]", res_db["type"])
                    print(turn["text"])
                    if domain in turn['metadata'].keys():
                        if "book" in turn['metadata'][domain].keys():
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

    origin_file1115 = dele_data["SNG1115.json"]
    for i, res_db in enumerate(db_file):
        if res_db["entrance fee"] != "?" and res_db["entrance fee"] != "free":
            filename1 = "SNG1115.json".upper().split('.')[0] + '_' + str(i)
            fname1 = filename1 + '.json'
            print(fname1)
            info1 = copy.deepcopy(origin_file1115)
            if info1["goal"][domain]:
                if "info" in info1["goal"][domain].keys():
                    for item in info1["goal"][domain]["info"].keys():
                        if info1["goal"][domain]["info"][item] != "dontcare" and info1["goal"][domain]["info"][item] != "not mentioned":
                            info1["goal"][domain]["info"][item] = res_db[item]
                if domain in info1["goal"][domain].keys():
                    info1["goal"][domain][domain] = res_db["name"]

                turns = info1["log"]
                for turn in turns:
                    print(turn["text"])
                    turn["text"] = turn["text"].replace("[" + domain + "_address]", res_db["address"])
                    turn["text"] = turn["text"].replace("[" + domain + "_name]", res_db["name"])
                    turn["text"] = turn["text"].replace("[" + domain + "_phone]", res_db["phone"])
                    turn["text"] = turn["text"].replace("[" + domain + "_postcode]", res_db["postcode"])
                    turn["text"] = turn["text"].replace("[value_area]", res_db["area"])
                    turn["text"] = turn["text"].replace("[value_count] pounds", res_db["entrance fee"])                    
                    turn["text"] = turn["text"].replace("[attraction_type]", res_db["type"])
                    print(turn["text"])
                    if domain in turn['metadata'].keys():
                        if "book" in turn['metadata'][domain].keys():
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

    origin_file1151 = dele_data["SNG1151.json"]
    for i, res_db in enumerate(db_file):
        if res_db["type"] == "boat":
            filename1 = "SNG1151.json".upper().split('.')[0] + '_' + str(i)
            fname1 = filename1 + '.json'
            print(fname1)
            info1 = copy.deepcopy(origin_file1151)
            if info1["goal"][domain]:
                if "info" in info1["goal"][domain].keys():
                    for item in info1["goal"][domain]["info"].keys():
                        if info1["goal"][domain]["info"][item] != "dontcare" and info1["goal"][domain]["info"][item] != "not mentioned":
                            info1["goal"][domain]["info"][item] = res_db[item]
                if domain in info1["goal"][domain].keys():
                    info1["goal"][domain][domain] = res_db["name"]

                turns = info1["log"]
                for turn in turns:
                    print(turn["text"])
                    turn["text"] = turn["text"].replace("[" + domain + "_address]", res_db["address"])
                    turn["text"] = turn["text"].replace("[" + domain + "_name]", res_db["name"])
                    turn["text"] = turn["text"].replace("[" + domain + "_phone]", res_db["phone"])
                    turn["text"] = turn["text"].replace("[" + domain + "_postcode]", res_db["postcode"])
                    turn["text"] = turn["text"].replace("[value_area]", res_db["area"])
                    turn["text"] = turn["text"].replace("[value_count] pounds", res_db["entrance fee"])                    
                    turn["text"] = turn["text"].replace("[attraction_type]", res_db["type"])
                    print(turn["text"])
                    if domain in turn['metadata'].keys():
                        if "book" in turn['metadata'][domain].keys():
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

    origin_file1151 = dele_data["SNG1151.json"]
    for i, res_db in enumerate(db_file):
        if res_db["type"] == "swimmingpool":
            filename1 = "SNG1151.json".upper().split('.')[0] + '_' + str(i)
            fname1 = filename1 + '.json'
            print(fname1)
            info1 = copy.deepcopy(origin_file1151)
            if info1["goal"][domain]:
                if "info" in info1["goal"][domain].keys():
                    for item in info1["goal"][domain]["info"].keys():
                        if info1["goal"][domain]["info"][item] != "dontcare" and info1["goal"][domain]["info"][item] != "not mentioned":
                            info1["goal"][domain]["info"][item] = res_db[item]
                if domain in info1["goal"][domain].keys():
                    info1["goal"][domain][domain] = res_db["name"]

                turns = info1["log"]
                for turn in turns:
                    print(turn["text"])
                    turn["text"] = turn["text"].replace("[" + domain + "_address]", res_db["address"])
                    turn["text"] = turn["text"].replace("[" + domain + "_name]", res_db["name"])
                    turn["text"] = turn["text"].replace("[" + domain + "_phone]", res_db["phone"])
                    turn["text"] = turn["text"].replace("[" + domain + "_postcode]", res_db["postcode"])
                    turn["text"] = turn["text"].replace("[value_area]", res_db["area"])
                    turn["text"] = turn["text"].replace("[value_count] pounds", res_db["entrance fee"])                    
                    turn["text"] = turn["text"].replace("go boating", "go swimming")
                    print(turn["text"])
                    if domain in turn['metadata'].keys():
                        if "book" in turn['metadata'][domain].keys():
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
write_data_for_internal_v1(dele_data, '/data/xyzhang/RL_research_folder/soloist-rl/soloist-cuhk/examples/multiwoz/data/standard/attraction/attraction5_constructednew.train.json', domain)

# write_data_for_internal_v1(full_data, '/home/xiaoying/soloist-cuhk/examples/multiwoz/data/standard/restaurant/restaurant45.valid_data.json', input_idx_data, '/home/xiaoying/soloist-cuhk/examples/multiwoz/data/standard/restaurant/restaurant45.valid_resnames.json')
# write_data_for_internal_v1(test_data, 'test.soloist.json', 'test.idx.json')