const navbar = document.querySelector("nav");

window.onscroll = () => {
	let topDistance = window.scrollY;
	if (topDistance >= 80) {
		navbar.classList.add("navbar-scroll");
	} else {
		navbar.classList.remove("navbar-scroll");
	}
};

// Navbar Hamburger Menu

const hamburgerMenu = document.querySelector(".hamburger");
const navbarLinks = document.querySelector(".navbar-links");

hamburgerMenu.addEventListener("click", () => {
	navbarLinks.classList.toggle("burger-active");
});
