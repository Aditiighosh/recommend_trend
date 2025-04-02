import prisma from './lib/db'; // Use ES module import for Prisma Client

async function main() {
  try {
    // Add a new user
    const newUser = await prisma.user.create({
      data: {
        email: 'alice@example.com', // User's email
        name: 'Alice',             // User's name
        posts: {
          create: [ // Create posts for this user
            {
              title: 'Alice\'s First Post',
              content: 'This is the content of Alice\'s first post.',
              published: true,
            },
            {
              title: 'Alice\'s Second Post',
              content: 'This is the content of Alice\'s second post.',
              published: false,
            },
          ],
        },
      },
    });
    console.log('Created User with Posts:', newUser);

    // Fetch all users with their posts
    const usersWithPosts = await prisma.user.findMany({
      include: { posts: true }, // Include related posts
    });
    console.log('All Users with Posts:', usersWithPosts);
  } catch (error) {
    console.error('Error:', error);
  } finally {
    await prisma.$disconnect();
  }
}

main();