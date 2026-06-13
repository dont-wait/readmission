# UI Analysis: Readmission Predictor

## 1. Thành phần tổng quan

| Thành phần | Đặc điểm / Chú trọng |
|------------|-----------------------|
| Header | Thanh header cố định, thiết kế phẳng, tối giản. Có vạch ngăn dọc và khoảng trắng rộng. |
| Title Section | Tiêu đề lớn "Vitals Entry" và phần mô tả nhỏ ngay bên dưới. |
| Form Grid | Lưới 3 cột đồng nhất cho các thẻ nhập liệu chỉ số sinh tồn. |
| Input Cards | Mỗi thẻ là một container độc lập với icon, nhãn, giá trị và đơn vị riêng. |
| Action Button | Nút bấm lớn, nổi bật, nằm ở giữa bên dưới form. |
| Data Footer | Dòng text bảo mật dữ liệu ở cuối trang. |

## 2. Thông số chi tiết (Design Tokens)

| Thành phần | Thông số |
|------------|----------|
| **Layout Structure** | Max-width: 1024px, centered. Background tổng thể: `#F8FAFF` |
| **Grid System** | 3 Column Grid, Gap: `20px` |
| **Padding** | Card-padding: `24px`, Container-padding: `48px` |
| **Typography** | Font: `Be Vietnam Pro`. H1: `36px/Bold`. Labels: `10px/Bold/Uppercase`. Values: `24px/Bold`. Units: `11px/Medium` |
| **Colors** | Background: `#F8FAFF`. Card: `#FFFFFF`. Text: `#001F3F` (Deep Navy). Accent: `#0047AB` (Royal Blue). Subtle: `#64748B` |
| **Borders** | Card-border: `1px solid #E2E8F0`. Border-radius: `16px` |
| **Shadows** | Card-shadow: `0 4px 6px -1px rgb(0 0 0 / 0.05)` |
| **Icons** | Lucide React (size: 16-24px). Color: `#0047AB` |
| **Interactive** | Button Hover: `#003366`. Input Focus: `box-shadow` & `border-color` |

## 3. Page Hierarchy

```
[Layout]
├── [Header]
│   ├── [Title + Separator]
│   └── [Profile/Status indicator]
├── [Main Content]
│   ├── [Title Section (H2 + P)]
│   ├── [Input Grid (3x3)]
│   │   └── [InputField Component x9]
│   ├── [Action Area]
│   │   ├── [Analyze Button]
│   │   └── [Security Label]
│   └── [Result Section (Conditional)]
└── [Footer]
```

## 4. Reusable Components

| Component | Props chính | Dùng ở đâu |
|-----------|-------------|------------|
| `InputField` | `label`, `unit`, `icon`, `value`, `onChange` | Trong grid nhập liệu |
| `Button` | `variant`, `isLoading`, `children`, `icon` | Nút chính và nút phụ |

## 5. Chiến lược bố cục & state

- **Layout:** Sử dụng CSS Grid cho Form (3 cột) và Flexbox cho các thành phần bên trong thẻ.
- **Styling:** Vì dự án không có Tailwind CSS thực sự, tôi sẽ triển khai các Class Utility cần thiết trong `global.css` mô phỏng Tailwind để code trong component gọn gàng hơn.
- **State:** Sử dụng React `useState` tại `PredictionPage` để quản lý `formData`.
