// 统一的页面初始化函数
function initializePage() {
    // 立即滚动到顶部
    window.scrollTo(0, 0);
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
    
    // 初始化拖拽功能
    initializeDragDrop();
    
    // 阻止自动聚焦
    preventAutoFocus();
}

function initializeDragDrop() {
    const slidingImages = document.querySelectorAll('.sliding-image');
    slidingImages.forEach((img) => {
        img.draggable = true;
        img.style.cursor = 'grab';
        
        img.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('text/plain', img.src);
            e.dataTransfer.setData('application/x-image-src', img.src);
            img.style.opacity = '0.5';
        });
        
        img.addEventListener('dragend', (e) => {
            img.style.opacity = '1';
        });
    });
}

function preventAutoFocus() {
    // 移除所有输入框的焦点
    document.querySelectorAll('input, textarea').forEach(el => {
        if (el.matches('input[type="text"], textarea')) {
            el.blur();
        }
    });
}

// 只使用一个事件监听器
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePage);
} else {
    initializePage();
}

// 确保页面加载完成后再次执行
window.addEventListener('load', () => {
    setTimeout(() => {
        window.scrollTo(0, 0);
        preventAutoFocus();
    }, 200);
});

/* ========== 让静态图片可拖进 Gradio Image 组件 ========== */
function enableStaticDragUpload() {
  console.log('开始初始化静态图片拖拽功能...');
  
  // ① 给每张静态图加可拖属性和视觉提示
  document.querySelectorAll('.horizontal-static-img').forEach((img, index) => {
    img.setAttribute('draggable', 'true');
    img.style.cursor = 'grab';
    
    img.addEventListener('dragstart', e => {
      console.log('开始拖拽图片:', img.src);
      e.dataTransfer.setData('text/plain', img.src);
      e.dataTransfer.effectAllowed = 'copy';
      img.style.opacity = '0.7';
    });
    
    img.addEventListener('dragend', e => {
      img.style.opacity = '1';
    });
  });

  // ② 等待并找到 Gradio Image 组件
  function findUploadComponents() {
    // 尝试多种选择器来找到上传组件
    const uploadCol = document.querySelector('#upload-col') || 
                     document.querySelector('[data-testid="upload-col"]') ||
                     document.querySelector('.gr-column:has(.gr-file-upload)') ||
                     document.querySelector('[data-elem-id="upload-col"]');
    
    if (!uploadCol) {
      console.log('未找到上传列容器，尝试直接找文件输入框...');
      // 直接查找文件输入框
      const uploadInput = document.querySelector('input[type="file"][accept*="image"]');
      if (uploadInput) {
        const uploadContainer = uploadInput.closest('.gr-file-upload') || uploadInput.parentElement;
        return { uploadCol: uploadContainer, uploadInput };
      }
      return null;
    }
    
    const uploadInput = uploadCol.querySelector('input[type="file"]') ||
                       uploadCol.querySelector('input[accept*="image"]');
    
    return { uploadCol, uploadInput };
  }

  // ③ 设置拖拽接收
  function setupDragReceiver() {
    const components = findUploadComponents();
    if (!components || !components.uploadInput) {
      console.log('未找到上传组件，3秒后重试...');
      setTimeout(setupDragReceiver, 3000);
      return;
    }

    const { uploadCol, uploadInput } = components;
    console.log('找到上传组件，设置拖拽接收...');

    // 防止默认行为
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      uploadCol.addEventListener(eventName, e => {
        e.preventDefault();
        e.stopPropagation();
      });
    });

    // 拖拽悬停效果
    uploadCol.addEventListener('dragover', e => {
      e.dataTransfer.dropEffect = 'copy';
      uploadCol.style.border = '3px dashed #4CAF50';
      uploadCol.style.background = 'rgba(76,175,80,0.1)';
    });

    uploadCol.addEventListener('dragleave', e => {
      // 确保真正离开了容器
      if (!uploadCol.contains(e.relatedTarget)) {
        uploadCol.style.border = '';
        uploadCol.style.background = '';
      }
    });

    // ④ 处理图片拖拽放置
    uploadCol.addEventListener('drop', async e => {
      console.log('检测到拖拽放置事件');
      uploadCol.style.border = '';
      uploadCol.style.background = '';
      
      const imageUrl = e.dataTransfer.getData('text/plain');
      if (!imageUrl) {
        console.log('未获取到图片URL');
        return;
      }

      console.log('准备上传图片:', imageUrl);

      try {
        // 使用代理方式或直接创建 blob URL
        const response = await fetch(imageUrl, {
          mode: 'cors',
          credentials: 'omit'
        }).catch(async (corsError) => {
          console.log('CORS错误，尝试使用代理:', corsError);
          // 如果 CORS 失败，尝试使用公共代理
          const proxyUrl = `https://api.allorigins.win/raw?url=${encodeURIComponent(imageUrl)}`;
          return fetch(proxyUrl);
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const blob = await response.blob();
        const fileName = imageUrl.split('/').pop()?.split('?')[0] || 'dragged-image.jpg';
        const file = new File([blob], fileName, { type: blob.type || 'image/jpeg' });

        // 创建新的 DataTransfer 对象并添加文件
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        
        // 设置到 input 并触发事件
        uploadInput.files = dataTransfer.files;
        
        // 触发多个事件确保 Gradio 检测到变化
        const events = ['change', 'input'];
        events.forEach(eventType => {
          const event = new Event(eventType, { bubbles: true, cancelable: true });
          uploadInput.dispatchEvent(event);
        });

        console.log('图片上传成功:', fileName);
        
        // 显示成功提示
        uploadCol.style.border = '3px solid #4CAF50';
        uploadCol.style.background = 'rgba(76,175,80,0.2)';
        setTimeout(() => {
          uploadCol.style.border = '';
          uploadCol.style.background = '';
        }, 2000);

      } catch (error) {
        console.error('图片上传失败:', error);
        
        // 显示错误提示
        uploadCol.style.border = '3px solid #f44336';
        uploadCol.style.background = 'rgba(244,67,54,0.1)';
        setTimeout(() => {
          uploadCol.style.border = '';
          uploadCol.style.background = '';
        }, 3000);
      }
    });
  }

  // 开始设置
  setupDragReceiver();
}

// 多重时机确保执行
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    setTimeout(enableStaticDragUpload, 1000);
  });
} else {
  setTimeout(enableStaticDragUpload, 1000);
}

window.addEventListener('load', () => {
  setTimeout(enableStaticDragUpload, 2000);
});