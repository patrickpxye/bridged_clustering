from cropnet.data_downloader import DataDownloader

# 1) initialize downloader
downloader = DataDownloader(target_dir="./data")

# 2) choose your target
target_year  = "2022"
target_fips  = ["10003"]       # e.g. county FIPS code
crop_name    = "Soybean"       # one of: "corn","cotton","Soybean","winter_wheat"

# 3) download exactly that USDA yield and AG imagery
downloader.download_USDA(
    crop_name,
    fips_codes=target_fips,
    years=[target_year]
)
downloader.download_Sentinel2(
    fips_codes=target_fips,
    years=[target_year],
    image_type="AG"
)
