import json, re

with open(r'C:\Users\skatkam\.gemini\antigravity-ide\brain\9f866c5a-a994-4834-816b-1bd4e0afbff1\.system_generated\steps\1091\content.md', 'r', encoding='utf-8') as f:
    text = f.read()

m = re.search(r'Box\.postStreamData\s*=\s*({.*?});', text, re.DOTALL)
if m:
    data = json.loads(m.group(1))
    shared_folder = data.get('/app-api/enduserapp/shared-folder', {})
    items = shared_folder.get('items', [])
    print(f"Found {len(items)} items in images folder:")
    for it in items:
        sz = it.get('itemSize', 0) / (1024 * 1024 * 1024)
        print(f"  {it.get('type')} id={it.get('id')} name={it.get('name')} size={sz:.2f} GB")
else:
    print('No postStreamData match')
