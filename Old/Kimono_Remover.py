import json
import random

with open('kimono_image_urls.json', 'r') as f:
    image_sources = json.load(f)
    
# unique_list = list(set(image_sources))
# print(len(unique_list))

# print(len(image_sources))

# with open('kimono_image_urls_remove.json', 'w') as f:
#     json.dump(unique_list, f)

def remove_duplicates_keep_order(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
        # else:
        #     if random.random() < 0.1:
        #         print(item)
    return result

# Example usage:
my_list = image_sources
unique_list = remove_duplicates_keep_order(my_list)

with open('kimono_image_urls_remove.json', 'w') as f:
    json.dump(unique_list, f)