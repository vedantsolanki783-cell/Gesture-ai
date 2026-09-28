import { MessageAttachment } from './hybridAI';

export async function parseUploadedFile(file: File): Promise<MessageAttachment> {
  const fileType = file.type;
  const fileName = file.name;

  // Handle plain text, code files (.ts, .py, .js, .json, .csv, .md, .txt)
  if (
    fileType.startsWith('text/') ||
    /\.(ts|tsx|js|jsx|py|json|csv|md|txt|html|css)$/i.test(fileName)
  ) {
    const content = await file.text();
    return { name: fileName, type: 'text', content };
  }

  // Handle Images (Convert to Base64 so vision models can read it)
  if (fileType.startsWith('image/')) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        resolve({
          name: fileName,
          type: 'image',
          content: reader.result as string,
        });
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  // Fallback for other documents
  const content = await file.text();
  return { name: fileName, type: 'text', content };
}
