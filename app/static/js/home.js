function submitForm() {
	const btn = document.querySelector("#submit-prompt");
	btn.classList.add("submit-loading");
	var formElement = document.getElementById('myForm');
	var data = new FormData(formElement);
	fetch('/submit', {
		method: 'POST',
		body: data,
	})
		.then(data => {
			document.getElementById("image").src="/static/images/generated.png?t=" + new Date().getTime();
			btn.classList.remove("submit-loading");
		})
		.catch(error => {
			console.error(error);
		});
}

function dynamic_grow(element) {
	element.style.height = "5px";
	element.style.height = (element.scrollHeight) + "px";
}

