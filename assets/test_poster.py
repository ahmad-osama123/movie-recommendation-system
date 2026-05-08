"""
تشخيص مشكلة الـ posters خطوة بخطوة
شغّله بـ: python3 test_poster.py
"""

import requests
from io import BytesIO

API_KEY = "17bb05940e431890b39c5ec1741771eb"

print("=" * 50)
print("STEP 1: Testing internet connection...")
try:
    r = requests.get("https://www.google.com", timeout=5)
    print(f"  ✅ Internet OK — status {r.status_code}")
except Exception as e:
    print(f"  ❌ No internet: {e}")
    exit()

print("\nSTEP 2: Testing TMDb API...")
try:
    r = requests.get(
        "https://api.themoviedb.org/3/search/movie",
        params={"api_key": API_KEY, "query": "Toy Story", "year": "1995"},
        timeout=8,
    )
    print(f"  Status code: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        results = data.get("results", [])
        print(f"  ✅ Got {len(results)} results")
        if results:
            first = results[0]
            print(f"  Title: {first.get('title')}")
            print(f"  poster_path: {first.get('poster_path')}")
        else:
            print("  ❌ Results list is empty!")
    elif r.status_code == 401:
        print("  ❌ API key غلط أو منتهي")
    elif r.status_code == 403:
        print("  ❌ Access denied — الـ IP ممكن يكون محجوب")
    else:
        print(f"  ❌ Unexpected status: {r.text[:200]}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    exit()

print("\nSTEP 3: Testing poster image download...")
try:
    poster_path = results[0].get("poster_path") if results else None
    if not poster_path:
        print("  ❌ No poster_path in results")
        exit()

    img_url = "https://image.tmdb.org/t/p/w300" + poster_path
    print(f"  URL: {img_url}")
    r2 = requests.get(img_url, timeout=8)
    print(f"  Status: {r2.status_code}")
    print(f"  Content-Type: {r2.headers.get('Content-Type')}")
    print(f"  Size: {len(r2.content)} bytes")

    if r2.status_code == 200 and len(r2.content) > 1000:
        print("  ✅ Image downloaded successfully!")

        # Test PIL
        try:
            from PIL import Image
            img = Image.open(BytesIO(r2.content))
            print(f"  ✅ PIL Image OK — size: {img.size}")
        except ImportError:
            print("  ⚠️ Pillow not installed — run: pip install Pillow")
        except Exception as e:
            print(f"  ❌ PIL Error: {e}")
    else:
        print("  ❌ Image download failed")

except Exception as e:
    print(f"  ❌ Error: {e}")

print("\nSTEP 4: Testing Streamlit version...")
try:
    import streamlit as st
    print(f"  ✅ Streamlit version: {st.__version__}")
except Exception as e:
    print(f"  ❌ {e}")

print("\n" + "=" * 50)
print("Done! Share the output above.")
