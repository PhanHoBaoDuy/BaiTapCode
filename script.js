// Đợi DOM load xong
document.addEventListener('DOMContentLoaded', function() {
const addButton = document.getElementById('add-task-btn');
const taskInput = document.getElementById('new-task-input');
const taskList = document.getElementById('task-list');
addButton.addEventListener('click', function() {
const taskText = taskInput.value;
if (taskText.trim() !== "") {
// Tạo một <li> mới
const newTask = document.createElement('li');
newTask.textContent = taskText;
// Thêm vào danh sách
taskList.appendChild(newTask);
// Xóa nội dung input
taskInput.value = "";
}
});
});