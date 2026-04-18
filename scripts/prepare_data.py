import json
import re
import random
from datasets import load_dataset

REFUSAL_RESPONSES = [
    "I'm sorry, but I can't assist with that. I am an offline assistant limited to weather, calendars, conversions, currency, and SQL.",
    "I don't have the capability to handle that request.",
    "As a local mobile assistant, I don't have a tool for that. I can help you check the weather or schedule a meeting, though!",
    "I cannot fulfill that request. My tools are strictly limited to weather, calendar, currency, conversions, and SQL queries.",
    "I am unable to process that. I am designed to operate offline for calendar management, conversions, weather, and SQL.",
    "That falls outside my supported features.",
    "Unfortunately, I can't do that. I'm a specialized offline agent."
]

CITIES = ["Tokyo", "New York", "London", "Paris", "Lahore", "Dubai", "Cairo", "Sydney", "Berlin", "Mumbai", "Toronto", "Riyadh"]
WEATHER_TEMPLATES = [
    ("What's the weather like in {city}?", "{unit}"),
    ("Tell me the temperature in {city}.", "{unit}"),
    ("Do I need an umbrella in {city} today?", "{unit}"),
    ("Give me the current forecast for {city}.", "{unit}"),
    ("Is it raining in {city} right now?", "{unit}")
]

SQL_TEMPLATES = [
    ("Get all users from the database.", "SELECT * FROM users;"),
    ("Find the email of the customer with ID {num}.", "SELECT email FROM customers WHERE id = {num};"),
    ("Count the number of orders placed.", "SELECT COUNT(*) FROM orders;"),
    ("Fetch the top 10 highest paid employees.", "SELECT * FROM employees ORDER BY salary DESC LIMIT 10;"),
    ("Show me the inventory for product {num}.", "SELECT stock FROM inventory WHERE product_id = {num};"),
    ("List all active subscriptions.", "SELECT * FROM subscriptions WHERE status = 'active';")
]

def bootstrap_missing_data(formatted_data, quotas, target):
    while quotas["weather"] < target:
        city = random.choice(CITIES)
        template, unit_pool = random.choice(WEATHER_TEMPLATES)
        unit = random.choice(["C", "F"])
        prompt = template.format(city=city)
        
        tool_call = {"tool": "weather", "args": {"location": city, "unit": unit}}
        formatted_data.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<tool_call>{json.dumps(tool_call)}</tool_call>"}]})
        quotas["weather"] += 1

    while quotas["sql"] < target:
        num = random.randint(1, 999)
        prompt_template, sql_template = random.choice(SQL_TEMPLATES)
        
        prompt = prompt_template.format(num=num)
        query = sql_template.format(num=num)
        
        tool_call = {"tool": "sql", "args": {"query": query}}
        formatted_data.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": f"<tool_call>{json.dumps(tool_call)}</tool_call>"}]})
        quotas["sql"] += 1

    return formatted_data, quotas

def build_dataset():
    print("Loading ungated open-source dataset (glaiveai/glaive-function-calling-v2)...")
    dataset = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    
    formatted_data = []
    quotas = {"weather": 0, "currency": 0, "calendar": 0, "convert": 0, "sql": 0, "refusals": 0}
    TARGET_PER_TOOL = 500 

    for item in dataset:
        if all(count >= TARGET_PER_TOOL for count in quotas.values()):
            break

        chat = item.get("chat", "")
        user_match = re.search(r"USER:\s*(.*?)\s*ASSISTANT:", chat, re.DOTALL)
        if not user_match:
            continue
        user_text = user_match.group(1).strip()
        
        func_match = re.search(r"<functioncall>\s*({.*?})", chat, re.DOTALL)
        
        if func_match:
            func_payload = func_match.group(1)
            name_match = re.search(r'"name":\s*"([^"]+)"', func_payload)
            args_match = re.search(r'"arguments":\s*(\'.*?\'|".*?")', func_payload)
            
            if not name_match:
                continue
                
            func_name = name_match.group(1).lower()
            func_args = {}
            if args_match:
                args_str = args_match.group(1)[1:-1] 
                try:
                    func_args = json.loads(args_str.replace('\\"', '"'))
                except json.JSONDecodeError:
                    pass

            if ("currency" in func_name or "exchange" in func_name) and quotas["currency"] < TARGET_PER_TOOL:
                amount = func_args.get("amount", 100)
                from_curr = str(func_args.get("from_currency", func_args.get("from", "USD")))[:3].upper() 
                to_curr = str(func_args.get("to_currency", func_args.get("to", "EUR")))[:3].upper()
                tool_call = {"tool": "currency", "args": {"amount": float(amount), "from": from_curr, "to": to_curr}}
                formatted_data.append({"messages": [{"role": "user", "content": user_text}, {"role": "assistant", "content": f"<tool_call>{json.dumps(tool_call)}</tool_call>"}]})
                quotas["currency"] += 1
                
            elif ("calendar" in func_name or "event" in func_name or "schedule" in func_name) and quotas["calendar"] < TARGET_PER_TOOL:
                date = func_args.get("date", "2026-04-19")
                title = func_args.get("title", func_args.get("event_name", "Meeting"))
                tool_call = {"tool": "calendar", "args": {"action": "create", "date": date, "title": title}}
                formatted_data.append({"messages": [{"role": "user", "content": user_text}, {"role": "assistant", "content": f"<tool_call>{json.dumps(tool_call)}</tool_call>"}]})
                quotas["calendar"] += 1

            elif ("convert" in func_name or "unit" in func_name or "distance" in func_name) and quotas["convert"] < TARGET_PER_TOOL:
                val = func_args.get("value", func_args.get("amount", func_args.get("distance", 10)))
                from_u = func_args.get("from_unit", "miles")
                to_u = func_args.get("to_unit", "km")
                tool_call = {"tool": "convert", "args": {"value": float(val), "from_unit": from_u, "to_unit": to_u}}
                formatted_data.append({"messages": [{"role": "user", "content": user_text}, {"role": "assistant", "content": f"<tool_call>{json.dumps(tool_call)}</tool_call>"}]})
                quotas["convert"] += 1
                
        else:
            if quotas["refusals"] < TARGET_PER_TOOL:
                if not any(word in user_text.lower() for word in ["weather", "convert", "currency", "calendar", "sql"]):
                    formatted_data.append({"messages": [{"role": "user", "content": user_text}, {"role": "assistant", "content": random.choice(REFUSAL_RESPONSES)}]})
                    quotas["refusals"] += 1
                    
        total_found = sum(quotas.values())
        if total_found > 0 and total_found % 500 == 0:
            print(f"Progress: Collected {total_found} natural examples...")

    print("\nBootstrapping missing weather and SQL data...")
    formatted_data, quotas = bootstrap_missing_data(formatted_data, quotas, TARGET_PER_TOOL)
            
    random.shuffle(formatted_data)

    with open("train_data.jsonl", "w") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + "\n")
            
    print("Dataset generation complete!")
    print("Final Quota Breakdown:")
    for key, count in quotas.items():
        print(f" - {key.capitalize()}: {count} examples")

if __name__ == "__main__":
    build_dataset()