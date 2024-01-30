import json
import random
from datetime import datetime, timedelta

def generate_date_list_for_year(year):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    delta = end_date - start_date
    return [start_date + timedelta(days=i) for i in range(delta.days + 1)]

def random_time():
    return datetime.now().replace(hour=random.randint(0, 23), 
                                 minute=random.randint(0, 59), 
                                 second=random.randint(0, 59), 
                                 microsecond=0)

def assign_dates_to_conversations(conversations_file_path, start_year, end_year):
    conversations = []
    with open(conversations_file_path, 'r') as file:
        for line in file:
            try:
                conversation = json.loads(line)
                conversations.append(conversation)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line.strip()} - Error: {e}")

    years = end_year - start_year + 1
    convos_per_year = len(conversations) // years
    extra_convos = len(conversations) % years

    index = 0
    for year in range(start_year, end_year + 1):
        date_list = generate_date_list_for_year(year)
        random.shuffle(date_list)
        num_convos_this_year = convos_per_year + (1 if extra_convos > 0 else 0)
        extra_convos -= 1

        for _ in range(num_convos_this_year):
            if index < len(conversations) and date_list:
                initial_date = date_list.pop()
                initial_time = random_time()
                initial_datetime = datetime.combine(initial_date, initial_time.time())
                time_increment = timedelta(seconds=random.randint(5, 15)) # Increment time by 5 to 15 seconds

                for message in conversations[index]:
                    # Format datetime and prepend it to message content
                    formatted_datetime = initial_datetime.strftime("%d:%m:%Y %H:%M:%S")
                    message['content'] = formatted_datetime + ' ' + message['content']
                    initial_datetime += time_increment
                index += 1

    new_file_path = conversations_file_path.replace('.jsonl', '_dated.jsonl')
    with open(new_file_path, 'w') as file:
        for convo in conversations:
            json.dump(convo, file)
            file.write('\n')
assign_dates_to_conversations('/home/ec2-user/environment/data_generation/character-create/data/sim_ns_role_duplicate_11724.jsonl', 2000, 2023)
