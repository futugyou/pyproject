import requests
import time
import json
import urllib.parse
import os

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://m.weibo.cn",
}


def strip_html(html):
    import re

    clean = re.compile("<.*?>")
    return re.sub(clean, "", html)


def get_full_text_and_region(mid):
    url = f"https://m.weibo.cn/statuses/show?id={mid}"
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            full_text = strip_html(data.get("text", ""))
            region_name = data.get("region_name", None)
            return full_text, region_name
    except Exception as e:
        print(f"Failed to get full text for {mid}: {e}")
    return None, None


def search_weibo(keyword, max_pages=3):
    encoded_keyword = urllib.parse.quote(keyword)
    containerid = f"100103type=1&q={encoded_keyword}"
    base_url = "https://m.weibo.cn/api/container/getIndex"

    results = []

    for page in range(1, max_pages + 1):
        params = {"containerid": containerid, "page_type": "searchall", "page": page}
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Request failed: {response.status_code}")
            break

        data = response.json()
        cards = data.get("data", {}).get("cards", [])

        for card in cards:
            if card.get("card_type") == 9:
                mblog = card.get("mblog", {})
                user = mblog.get("user", {})
                mid = mblog.get("mid", "")
                short_text = strip_html(mblog.get("text", ""))
                is_long = mblog.get("isLongText", False)

                full_text, region_name = (None, None)
                if is_long or "...全文" in short_text:
                    full_text, region_name = get_full_text_and_region(mid)
                    time.sleep(0.5)

                result = {
                    "weibo_id": mid,
                    "weibo_url": f"https://m.weibo.cn/detail/{mid}",
                    "text": full_text or short_text,
                    "created_at": mblog.get("created_at", ""),
                    "user": user.get("screen_name", ""),
                    "user_id": user.get("id", ""),
                    "attitudes_count": mblog.get("attitudes_count", 0),
                    "comments_count": mblog.get("comments_count", 0),
                    "reposts_count": mblog.get("reposts_count", 0),
                    "region_name": region_name,  # 可为 None
                }

                results.append(result)

        time.sleep(1)

    return results


def load_existing_ids(filepath):
    if not os.path.exists(filepath):
        return set()

    ids = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                ids.add(item.get("weibo_id"))
            except:
                continue
    return ids


def save_as_jsonl_incremental(data, filename="weibo_foodsafety.jsonl"):
    existing_ids = load_existing_ids(filename)
    new_entries = [entry for entry in data if entry["weibo_id"] not in existing_ids]

    with open(filename, "a", encoding="utf-8") as f:
        for entry in new_entries:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + "\n")

    print(
        f"Found {len(data)} total, {len(new_entries)} new entries saved to {filename}"
    )


if __name__ == "__main__":
    keyword = "食品安全"
    data = search_weibo(keyword, max_pages=5)
    save_as_jsonl_incremental(data, filename="1.weibo_foodsafety.jsonl")
