
import oss2, sys

import back_configuration as bc

sys.stdout.reconfigure(encoding='utf-8')

# 将上传的图片路径储存到OSS对应的bucket中，然后转换为URL
def file_paths_oss_url(access_key_id, access_key_secret, bucket_name, oss_endpoint, local_img_path):
    # 上传
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, oss_endpoint, bucket_name)

    # 将Windows路径分隔符转换为正斜杠，用于OSS对象名
    oss_object_name = local_img_path.replace('\\', '/')
    with open(local_img_path, 'rb') as fileobj:
        bucket.put_object(oss_object_name, fileobj)

    # 拼接公网URL
    oss_url = f'https://{bucket_name}.{oss_endpoint}/{oss_object_name}'
    return oss_url


if __name__ == "__main__":
    local_img_path = r'D:\桌面\second_sky_hackathon\images\uploaded_20250708_215208_f90973ff.png'
    access_key_id, access_key_secret, bucket_name, oss_endpoint = bc.img_to_oss_url()

    # 没有其他意思，只做测试使用的
    print("存储放在OSS上的图片URL为：",file_paths_oss_url(access_key_id, access_key_secret, bucket_name, oss_endpoint, local_img_path))
