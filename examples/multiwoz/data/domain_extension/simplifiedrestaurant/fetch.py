import json,copy
full_data = json.load(open('/Users/zxy/Desktop/simplifiedrestaurant/restaurant_domain_extension10_simplified_forsoloist.json'))
idx_file = json.load(open('restaurant_domain_extension5.idx.json'))

examples = {}
for fname, fcontent in full_data.items():
    if fname in idx_file:
        examples[fname] = full_data[fname]

# json.dump(examples, open('restaurant_domain_extension5noisy.json','w'),indent=2)

def write_data_for_internal_v1(data, output_file, output_idx_file):
    flist = []
    examples = []
    for fname, info in data.items():
        history = []
        example = {}
        for turn in info:
            user = turn['usr_no_delex'].strip().lower()
            # "usr_no_delex": "am looking for a place to to stay that has cheap price range it should be in a type of hotel"
            system_nl = turn['sys'].strip().lower()
            # "sys": "okay , do you have a specific area you want to stay in ?"
            system_no_delex_nl = turn['sys_no_delex'].strip().lower()
            # "sys_no_delex": "Okay, do you have a specific area you want to stay in?"
            # kb = turn['db'].lower()
            # kb = turn['db']
            ds = turn['bs']
            # "bs": [
            #     "hotel pricerange = cheap ; type = hotel"
            # ]
            dp = 'dp : ' + turn['dp'][0]
            #             "dp": [
            #     "Hotel ( Request ( Area ) )"
            # ]
            dp = dp.lower()
            # active_domains = [i.split()[0] for i in ds] # active_domains = ['hotel']
            # i = 'hotel pricerange = cheap ; type = hotel'
            # i.split() = ['hotel', 'pricerange', '=', 'cheap', ';', 'type', '=', 'hotel']
            
            # kb_nums = kb.split('|')[0].split(';')
            # kb_nums_dict = dict([i.strip().split(' : ') for i in kb_nums])
            # for active_domain in active_domains:
            #     if active_domain not in ['none','taxi','hospital']:
            #         kb_nums = int(kb_nums_dict[active_domain])
            #         # multiwoz style kb feature
            #         if active_domain != 'train':
            #             if kb_nums > 5:
            #                 kb_nums = 'more than five'
            #             elif kb_nums == 0:
            #                 kb_nums = 'zero'
            #             elif kb_nums == 1:
            #                 kb_nums = 'one'
            #             elif kb_nums == 2:
            #                 kb_nums = 'two'
            #             elif kb_nums == 3:
            #                 kb_nums = 'three'
            #             elif kb_nums == 4:
            #                 kb_nums = 'four'
            #         else:
            #             if kb_nums > 40:
            #                 kb_nums = 'more than five'
            #             elif kb_nums == 0:
            #                 kb_nums = 'zero'
            #             elif kb_nums <= 2:
            #                 kb_nums = 'one'
            #             elif kb_nums <= 5:
            #                 kb_nums = 'two'
            #             elif kb_nums <= 10:
            #                 kb_nums = 'three'
            #             elif kb_nums <= 40:
            #                 kb_nums = 'four'
            #         kb = f'kb : {active_domain} {kb_nums}'
            #     else:
            #         kb = f'kb : {active_domain}'
            
            history.append(f'user : {user}')

            user = f'user : {user} '
            # if len(ds) == 0:
            #     ds = 'none'
            # else:
            #     ds = ' | '.join(ds)
            if len(ds) == 0:
                ds = 'none'
            else:
                ds = ds[0]
            ds = f'belief : {ds}'.lower()
            sys = f'system : {system_nl}'
            # system_nl is delex. sys. response
            example['history'] = copy.copy(history)
            # example['kb'] = kb
            example['belief'] = ds
            example['reply'] = sys # delex. sys. response
            example['name'] = fname
            example['dp'] = dp
            examples.append(copy.deepcopy(example))
            history.append(sys)
            flist.append(fname)

    json.dump(flist, open(output_idx_file,'w'))
    json.dump(examples, open(output_file,'w'),indent=2)
    print(len(examples))
# len(train.soloist.json) = 56678
# len(valid.soloist.json) = 7374
# len(test.soloist.json) = 7372
# write_data_for_internal_v1(train_data, 'data/train_sampled.soloist.json', 'data/train_sampled.idx.json')
# write_data_for_internal_v1(valid_data, 'data/valid_sampled.soloist.json', 'data/valid_sampled.idx.json')
write_data_for_internal_v1(examples, '/Users/zxy/Desktop/simplifiedrestaurant/restaurant_domain_extension5_simplified_reward.soloist.json', '/Users/zxy/Desktop/simplifiedrestaurant/restaurant_domain_extension5_simplified_reward.idx.json')