""" 利用阿里云皮肤病分析模型进行视觉性地采集皮肤上的一些表征数据
目前，阿里云推出的是公测版，基本没有成本，但是对速率和并发量有限制

"""
from typing import List
import json
from alibabacloud_imageprocess20200320.client import Client as imageprocess20200320Client
from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_imageprocess20200320 import models as imageprocess_20200320_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient

import back_configuration as bc
import sys
sys.stdout.reconfigure(encoding='utf-8')

class Sample:
    @staticmethod
    def create_client(skin_analysis) -> imageprocess20200320Client:
        config = open_api_models.Config(
            access_key_id=skin_analysis.get('access_key_id'),
            access_key_secret=skin_analysis.get('access_key_secret'),
        )
        config.endpoint = skin_analysis.get('endpoint', 'imageprocess.cn-shanghai.aliyuncs.com')
        return imageprocess20200320Client(config)

    @staticmethod
    def main(
        args: List[str],
        skin_analysis,
        oss_img_url
    ) -> None:
        client = Sample.create_client(skin_analysis)
        detect_skin_disease_request = imageprocess_20200320_models.DetectSkinDiseaseRequest(
            url=oss_img_url,
            org_id=skin_analysis.get('org_id'),
            org_name=skin_analysis.get('org_name')
        )
        runtime = util_models.RuntimeOptions()
        try:
            response = client.detect_skin_disease_with_options(detect_skin_disease_request, runtime)
            # 格式化输出
            def obj_to_dict(obj):
                if isinstance(obj, dict):
                    return {k: obj_to_dict(v) for k, v in obj.items()}
                elif hasattr(obj, '__dict__'):
                    return {k: obj_to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
                elif isinstance(obj, list):
                    return [obj_to_dict(i) for i in obj]
                else:
                    return obj

            # 在打印前使用
            skin_analysis_data = json.dumps(obj_to_dict(response.body.data), ensure_ascii=False, indent=2)

            return skin_analysis_data
        except Exception as error:
            print(getattr(error, 'message', str(error)))
            if hasattr(error, 'data') and error.data:
                print(error.data.get("Recommend"))
            UtilClient.assert_as_string(getattr(error, 'message', str(error)))


if __name__ == '__main__':
    local_img_path = r'D:\桌面\second_sky_hackathon\images\uploaded_20250708_215208_f90973ff.png'

    # 实例化测试配置
    skin_analysis, oss_img_url = bc.skin_analysis_instantiation(local_img_path)
    # 得到分析结果
    result = Sample.main(sys.argv[1:], skin_analysis, oss_img_url)
    # 打印结果
    print(result)