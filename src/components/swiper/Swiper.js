const swiper = new Swiper(".mySwiper", {
  spaceBetween: 5, // jarak antar slide lebih rapat
  loop: true,

  navigation: {
    nextEl: ".swiper-button-next",
    prevEl: ".swiper-button-prev",
  },

  // Pengaturan agar slider responsif
  breakpoints: {
    640: {
      slidesPerView: 1,
      spaceBetween: 8,
    },
    768: {
      slidesPerView: 2,
      spaceBetween: 10,
    },
    1024: {
      slidesPerView: 3,
      spaceBetween: 10,
    },
    1440: {
      slidesPerView: 4,
      spaceBetween: 12,
    },
  },
});
