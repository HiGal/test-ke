import s3fs


if __name__ == '__main__':

    # Подключение до S3
    fs = s3fs.S3FileSystem(key="oGG7onz8X7dfT_AKvwKn",
                           secret="YT74LUkrzipAFWcBmHuBOszIxwdLFtjNPkxkVV4O",
                           client_kwargs={'endpoint_url': "http://storage.yandexcloud.net"})

    with fs.open(f"nlp-test-task/data.csv", "rb") as remote_file:
        # Скачает датасет в файл data.csv
        with open("data.csv", "wb") as local_file:
            local_file.write(remote_file.read())