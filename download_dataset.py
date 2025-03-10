import os
import subprocess

# Create dataset directories
os.makedirs("dataset/movie_videos", exist_ok=True)
os.makedirs("dataset/real_life_videos", exist_ok=True)

# Movie Dataset URLs (You can add more movie clip URLs)
movie_urls = [
    "https://youtube.com/shorts/eNkl0KP3maQ?si=oq_9v7kUdCd79ySM",
    "https://youtube.com/shorts/WfK3JoTeKdA?si=cI8SBCknrHNAJSPs",
    "https://youtube.com/shorts/WIz9FSDrINE?si=JR6mn7rYxggcsUnC",
    "https://youtube.com/shorts/4TsuFHC6Z3k?si=GZPZbPPrk5yedjwo",
    "https://youtube.com/shorts/GDjIKRl2u74?si=5lz9GphVf1DQvPO0",
    "https://youtube.com/shorts/gQsGjLwIT6o?si=H2DfvPstcab3twvd",
    "https://youtube.com/shorts/xcn5r7feO60?si=EO749wz8uJpoi4Q7",
    "https://youtube.com/shorts/PLYQkFSWu2A?si=PkCJo0DQwowknKow",
    "https://youtube.com/shorts/LeGHQThnTnc?si=vzXKGdCXQ8dXTDIw",
    "https://youtube.com/shorts/_ocXtT9UATU?si=vHvJDO60mvkSmkgD",
    "https://youtube.com/shorts/IOlPh-GFKI8?si=z9ZN1JmInxi6vCGR",
    "https://youtube.com/shorts/Dthh2EVkABw?si=-uUzy24brV_UsqeW",
    "https://youtube.com/shorts/U3gnCka1ASw?si=WL8PWGhHGbef72YF",
    "https://youtube.com/shorts/Dov-CwTQpBI?si=S0KLj55jB9bxoVRZ",
    "https://youtu.be/1CDlBLvc3YE?si=AeEPp7DDvu3izM7c",
    "https://youtu.be/C--iuM8NcQc?si=0UcxbtoHw89U3ILx",
    "https://youtu.be/E84VqqCPI7w?si=e4ut8OfaVxvwdoKU",
    "https://youtu.be/lQkpes3dgzg?si=s3_0p0PQA62cmi_b",
    "https://youtu.be/lQkpes3dgzg?si=fuxyDMrbwh7QnMyy",
    "https://youtu.be/OIBpHO1gZgQ?si=8iSTFHjChCBLfx9j",
    "https://youtu.be/GyR4RK0LA_E?si=AzON2QkIa4F3Z51Z",
    "https://youtu.be/IeL8EYtbVw0?si=hIQq7bMKgAcJdibw",
    "https://youtu.be/X7LsBMA-rKg?si=o9ZzvtcblLTqWR5z",
    "https://youtu.be/RgDPi5WvC8M?si=amveObI7-KXuZqqQ",
    "https://youtu.be/aRM2YcGpmxg?si=tN979j3P3sDaHatS",
    "https://youtu.be/PSZxmZmBfnU?si=MIAqjT35pK_RyEjh",
    "https://youtu.be/0jxVnlRdelU?si=1k1o_MNFbBc-0SCu",
    "https://youtu.be/ZcQtUdZ5Afs?si=tncuSyYxrvvWQmHz",
    "https://youtu.be/dC1yHLp9bWA?si=8zEjEysDmRcRlJnk",
    "https://youtu.be/gwdypLFy8Pk?si=IzYYlls09KejpEah",
    "https://youtu.be/J7GY1Xg6X20?si=P_goE4pbQAo1rKjz",
    "https://youtu.be/35D-9K3IyyM?si=k8wgabOYYnApr1VP",
    "https://youtu.be/14pdNYXY3Zo?si=rq9nbafwwV3cKSY0",
    "https://youtu.be/w40ushYAaYA?si=g1ki4jPjOTR3vGd9",
    "https://youtu.be/w40ushYAaYA?si=2tEgYeiY7wxSpD4w",
    "https://youtu.be/0zHmeTeLgMY?si=Hz_-EeWmZiR9WhIw",
    "https://youtu.be/JehjqlzXwIQ?si=gP_w_HwqoEoh-4EY",
    "https://youtu.be/Jh88x9Ejd5U?si=48QwNKc-HnEIoIr-",
    "https://youtu.be/UYa6gbDcx18?si=mx-iYgOTSviMAREc",
    "https://youtu.be/GQ5ICXMC4xY?si=xJ9bnQPsiGyAUJBJ",
    "https://youtu.be/GQ5ICXMC4xY?si=hDyFMCh_oa2iVBQD",
    "https://youtu.be/99Ptctl5_qQ?si=4asP0DSJwGpDD0hm",
    "https://youtu.be/wtpOtFIEkbs?si=FBppb4TU2cqSWPmz",
    "https://youtu.be/cfxJCdBFuLk?si=DnIDBTh0IbIdhrj5",
    "https://youtu.be/zE7PKRjrid4?si=3oymS_jmXFsqHtkY",
    "https://youtu.be/vPrPfkT82Pw?si=qa1av01lDJxqsxRq",
    "https://youtu.be/HOocTXKPVVU?si=Gt4YSJlYfQBuzVes",
    "https://youtu.be/GZR58d77a4A?si=H0vXRhDQ0bO4XAif",
    "https://youtu.be/NLvKtstNUD0?si=r1WxO_jjQPZ2SB_T",
    "https://youtu.be/sdrTKPyW018?si=2NYzwjOvCyKKnnY-",
    "https://youtu.be/HMiHoI5d8Po?si=01lzkgRWqTyoEW_q",
    "https://youtu.be/wxN2Mewamj0?si=XQyK7ZWz9i_cfkXV",
    "https://youtu.be/mJZZNHekEQw?si=C-A7JBXof5Ud8JHa",
    "https://youtu.be/yHqdESArkqU?si=bpBfTi8DJ15ryB1W",
    "https://youtu.be/jZOywn1qArI?si=rbHcLZXUO-wnwU-B",
    "https://youtu.be/OLdIKlXl3ZA?si=nQu7LcrCep6nJngl",
    "https://youtu.be/82RTzi5Vt7w?si=nqonJWHKhzN3-ZFj",
    "https://youtu.be/79p57JJrMwo?si=p1bZziLjH8Tj3L0Y",
    "https://youtu.be/gcn6v7IaIfA?si=0LaeZs5uw1JEJgn2",
    "https://youtu.be/WgJ3WqCRuKg?si=3WRUs_x6QT_uVJNw",
    "https://youtu.be/25CtoSJD9eo?si=dKG9hM8MulFtp360",
    "https://youtu.be/25CtoSJD9eo?si=JFUXIxB13QnNA7zI",
    "https://youtu.be/5i0u4jFmE78?si=Q6QBZg9Tx_h6j-SM",
    "https://youtu.be/ax7wcShvrus?si=ohmIpJ47rVqLkzV0",
    "https://youtu.be/2Tzg_S4NrRQ?si=3lSJhVuBCjz3QoB7",
    "https://youtu.be/zZ5gCGJorKk?si=USovZ4bdMq_E2tfo",
    "https://youtu.be/8Xjr2hnOHiM?si=aFDO1fcaj2tfJxcR"
    "https://youtu.be/35fLKn2Tq3o?si=pmYyQZmd71E6hcfj",
    "https://youtu.be/j51DfrLHUek?si=C5dNiLS5AyjxNdgn",
    "https://youtu.be/BX-FMvt83fA?si=v6Gok7ZGZYbum8Ff",
    "https://youtu.be/wuk8AOjGURE?si=nIBZti5fB--YMAMF",
    "https://youtu.be/WDpipB4yehk?si=MBlpdTxULZ5TePur",
    "https://youtu.be/QUYKSWQmkrg?si=QtcWLcYacpP9BQhw",
    "https://youtu.be/uYWQAg12Ko0?si=Jk30Ffw9Q0R5YYqC",
    "https://youtu.be/WI7ePVquOtk?si=eTFvAbZ71IIc1DU9",
    "https://youtu.be/1gpXMGit4P8?si=lNgNYUtcpT3t2Lzh",
    "https://youtu.be/aBpwrORhKWU?si=ZRgcc-2E6I42JtMa",
    "https://youtu.be/QNahH5nQqHs?si=QKlWFBgWjtwZSb6a",
    "https://youtu.be/EJR1H5tf5wE?si=OWCRhA9TTXq_AOGW",
    "https://youtu.be/A3oL7v7PLac?si=2cqsT2c0pTiHT8jo",
    "https://youtube.com/shorts/WNxpX4vgRVk?si=8f1qhgcZa-Mc_wyv",
    "https://youtube.com/shorts/8VoKpxzdnqk?si=FA56f0-lw9Gdxsng",
    "https://youtube.com/shorts/wNrhQst6XDY?si=uc96g7_48ddQL30P",
    "https://www.youtube.com/watch?v=ZUx1t5Tf2hA",
    "https://www.youtube.com/watch?v=ZUx1t5Tf2hA",
    "https://www.youtube.com/watch?v=ZUx1t5Tf2hA",
    "https://www.youtube.com/watch?v=rMPkUuMq024",
    "https://www.youtube.com/watch?v=0Taa1-Exd8I",
    "https://www.youtube.com/watch?v=uc0Hh-IrqL4",
    "https://www.youtube.com/watch?v=ZUx1t5Tf2hA",
    "https://www.youtube.com/watch?v=RhD5qUUzX1Q",
    "https://www.youtube.com/watch?v=ZUx1t5Tf2hA",
    "https://www.youtube.com/watch?v=gpKqDR9ZEyU" 
]

# Function to download videos
def download_videos(urls, save_path):
    for url in urls:
        command = f'python -m yt_dlp -f "best" -o "{save_path}/%(title)s.%(ext)s" {url}'
        subprocess.run(command, shell=True)

# Download datasets
download_videos(movie_urls, r"D:\video_classification_mannually\video_classification_without_any_model\dataset\movie_videos")

print("✅ All videos downloaded successfully!")
   